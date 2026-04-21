import asyncio, base64, cv2, httpx, io, json, logging, os, queue, time
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image
from threading import Thread, Lock
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VLLM_URL       = os.getenv("VLLM_URL",   "http://qwen3vl-30b-awq:8000/v1")
MODEL_NAME     = os.getenv("MODEL_NAME", "Qwen3-VL-30B-A3B-Instruct-AWQ")
FRAME_INTERVAL = float(os.getenv("FRAME_INTERVAL", "2.0"))

SAFETY_PROMPT = """Analyze this warehouse/factory image for safety hazards.
Return ONLY valid JSON:
{"score":<int 0-100>,"deductions":[{"type":"<hazard>","points":<negative int>,"detail":"<brief>"}],"summary":"<one sentence>"}
Scoring deductions (each category max once):
PPE violation:-10, Blocked Exit/Fire Hazard:-30, Stacking/Load Safety:-25, Trip Hazard:-20, Forklift Hazard:-15, Spill/Chemical Hazard:-35
Only report clearly visible hazards. Return score 100 with empty deductions if safe."""

# ── State ──────────────────────────────────────────────────────────────────────
state = {
    "cap": None, "running": False, "loop_video": True,
    "current_frame_b64": None, "last_analysis": None,
    "alert_prompt": None, "alert_threshold": 40, "alert_active": False, "webhook_url": None,
    "chat_prompt": None, "vlm_busy": False,
}
state_lock = Lock()
ws_clients: list[WebSocket] = []
broadcast_queue: queue.Queue = queue.Queue()


# ── VLM helpers ────────────────────────────────────────────────────────────────
def encode_frame(frame: np.ndarray) -> str | None:
    h, w = frame.shape[:2]
    if h < 28 or w < 28:  # reject frames too small for vision processor
        return None
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img).resize((336, 336))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def call_vlm_sync(prompt: str, b64: str) -> str:
    messages = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        {"type": "text", "text": prompt},
    ]}]
    try:
        r = httpx.post(f"{VLLM_URL}/chat/completions",
                       json={"model": MODEL_NAME, "messages": messages,
                             "max_tokens": 512, "temperature": 0.1},
                       timeout=60.0)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"VLM error: {e}"


def parse_safety_json(raw: str) -> dict:
    raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
    try:
        return json.loads(raw)
    except Exception:
        return {"score": -1, "deductions": [], "summary": raw[:200]}


# ── Frame loop (background thread — never touches asyncio) ─────────────────────
def frame_loop():
    last_vlm_time = 0.0
    while True:
        with state_lock:
            running = state["running"]
            cap = state["cap"]
        if not running or cap is None:
            time.sleep(0.1)
            continue

        ret, frame = cap.read()
        if not ret:
            with state_lock:
                if state["loop_video"]:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    state["running"] = False
                    break

        _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        frame_b64 = base64.b64encode(jpg.tobytes()).decode()
        with state_lock:
            state["current_frame_b64"] = frame_b64

        # Push frame to broadcast queue (asyncio picks it up)
        broadcast_queue.put({"type": "frame", "b64": frame_b64})

        now = time.time()
        with state_lock:
            busy = state["vlm_busy"]

        if not busy and (now - last_vlm_time) >= FRAME_INTERVAL:
            with state_lock:
                state["vlm_busy"] = True
                alert_prompt = state["alert_prompt"]
                chat_prompt = state["chat_prompt"]
                state["chat_prompt"] = None
                threshold = state["alert_threshold"]

            # Always use safety prompt for scoring; alert is just a threshold check on the score
            prompt = SAFETY_PROMPT
            is_alert = False
            is_safety = True
            vlm_b64 = encode_frame(frame)
            if vlm_b64 is None:
                with state_lock:
                    state["vlm_busy"] = False
                last_vlm_time = now
                continue

            def run_vlm(p=prompt, b=vlm_b64, ia=is_alert, iss=is_safety, thr=threshold):
                raw = call_vlm_sync(p, b)
                result = {"type": "vlm", "prompt": p, "raw": raw,
                          "is_alert": ia, "is_safety": iss}
                if iss or ia:
                    parsed = parse_safety_json(raw)
                    result["analysis"] = parsed
                    with state_lock:
                        state["last_analysis"] = parsed
                        alert_on = state["alert_active"]
                    score = parsed.get("score", 100)
                    if alert_on and score != -1 and score <= thr:
                        result["alert_triggered"] = True
                        result["alert_score"] = score
                        with state_lock:
                            webhook = state.get("webhook_url")
                        if webhook:
                            try:
                                import httpx as _httpx
                                _httpx.post(webhook, json={"score": score, "hazards": [d["type"] for d in parsed.get("deductions", [])], "summary": parsed.get("summary","")}, timeout=5.0)
                            except Exception as _e:
                                logger.warning(f"Webhook failed: {_e}")
                with state_lock:
                    state["vlm_busy"] = False
                broadcast_queue.put(result)

            Thread(target=run_vlm, daemon=True).start()
            last_vlm_time = now

        time.sleep(1 / 30)


# ── Async broadcast pump (runs in FastAPI event loop) ─────────────────────────
async def broadcast_pump():
    while True:
        try:
            msg = broadcast_queue.get_nowait()
            if msg.get("type") == "vlm":
                logger.info(f"Broadcasting VLM result: score={msg.get('analysis', {}).get('score')}")
            dead = []
            for ws in ws_clients:
                try:
                    await ws.send_json(msg)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                ws_clients.remove(ws)
        except queue.Empty:
            pass
        await asyncio.sleep(0.033)  # ~30fps pump


@asynccontextmanager
async def lifespan(app: FastAPI):
    Thread(target=frame_loop, daemon=True).start()
    asyncio.ensure_future(broadcast_pump())
    yield

app = FastAPI(title="SafetyLens v2", lifespan=lifespan)


# ── API endpoints ──────────────────────────────────────────────────────────────
@app.post("/api/upload")
async def upload_video(video: UploadFile = File(...), loop: bool = True):
    import time as _t
    _t0 = _t.time()
    data = await video.read()
    path = f"/tmp/{video.filename}"
    with open(path, "wb") as f:
        f.write(data)
    logger.info(f"Upload {video.filename}: {len(data)/1024/1024:.1f}MB in {_t.time()-_t0:.2f}s")
    with state_lock:
        if state["cap"]:
            state["cap"].release()
        state["cap"] = cv2.VideoCapture(path)
        state["video_path"] = path
        state["loop_video"] = True  # always loop uploaded videos
        state["running"] = True
        state["last_analysis"] = None
    return {"status": "started", "file": video.filename}


@app.post("/api/rtsp")
async def set_rtsp(request: Request):
    body = await request.json()
    logger.info(f"RTSP/path request: {body}")
    url = body.get("url", "")
    import asyncio, concurrent.futures
    loop = asyncio.get_event_loop()
    def open_cap():
        cap = cv2.VideoCapture(url)
        return cap if cap.isOpened() else None
    cap = await loop.run_in_executor(None, open_cap)
    if cap is None:
        return JSONResponse({"error": f"Cannot open: {url}"}, status_code=400)
    with state_lock:
        if state["cap"]:
            state["cap"].release()
        state["cap"] = cap
        state["loop_video"] = True
        state["running"] = True
        state["last_analysis"] = None
    return {"status": "started", "url": url}


@app.post("/api/stop")
async def stop():
    with state_lock:
        state["running"] = False
        if state["cap"]:
            state["cap"].release()
            state["cap"] = None
    return {"status": "stopped"}




@app.post("/api/alert")
async def set_alert(request: Request):
    body = await request.json()
    active = body.get("active", False)
    with state_lock:
        state["alert_active"] = active
        state["alert_prompt"] = body.get("prompt") if active else None
        state["alert_threshold"] = int(body.get("threshold", 40))
        state["webhook_url"] = body.get("webhook_url") or None
    return {"status": "ok"}


@app.get("/api/status")
async def status():
    with state_lock:
        return {"running": state["running"], "last_analysis": state["last_analysis"],
                "alert_active": state["alert_active"]}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    ws_clients.append(ws)
    with state_lock:
        frame = state["current_frame_b64"]
        analysis = state["last_analysis"]
    if frame:
        await ws.send_json({"type": "frame", "b64": frame})
    if analysis:
        await ws.send_json({"type": "vlm", "is_safety": True, "analysis": analysis,
                            "raw": analysis.get("summary", ""), "prompt": SAFETY_PROMPT})
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        if ws in ws_clients:
            ws_clients.remove(ws)


@app.get("/", response_class=HTMLResponse)
async def ui():
    return r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SafetyLens v2</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0a0e1a;color:#e2e8f0;height:100vh;overflow:hidden;display:flex;flex-direction:column}
.header{background:linear-gradient(135deg,#0f172a,#1e293b);padding:12px 24px;border-bottom:1px solid #334155;display:flex;align-items:center;gap:12px;flex-shrink:0}
.header h1{font-size:1.3rem;color:#f59e0b}
.header span{color:#94a3b8;font-size:1.1rem}
.main{display:grid;grid-template-columns:1fr 400px;flex:1;min-height:0}
.left{display:flex;flex-direction:column;padding:12px;gap:10px;min-height:0}
.video-wrap{position:relative;background:#000;border-radius:10px;overflow:hidden;flex:1;min-height:0}
.video-wrap img{width:100%;height:100%;object-fit:contain}
.video-overlay{position:absolute;top:8px;left:8px;display:flex;gap:6px;flex-wrap:wrap}
.badge{padding:6px 14px;border-radius:6px;font-size:2rem;font-weight:900;text-shadow:0 2px 8px #000}
.badge-green{background:#22c55e30;color:#4ade80;border:1px solid #22c55e}
.badge-yellow{background:#eab30830;color:#fde047;border:1px solid #eab308}
.badge-red{background:#ef444430;color:#fca5a5;border:1px solid #ef4444;animation:pulse 1s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.6}}
.score-big{position:absolute;top:8px;right:8px;font-size:2rem;font-weight:900;text-shadow:0 2px 8px #000}
.alert-overlay{display:none;position:absolute;inset:0;background:#ef444440;align-items:center;justify-content:center;z-index:10;animation:pulse 0.5s infinite}
.alert-overlay .alert-icon{font-size:8rem;filter:drop-shadow(0 0 20px #ef4444)}
.hazard-tags{position:absolute;bottom:8px;left:8px;display:flex;gap:4px;flex-wrap:wrap}
.tag{padding:4px 12px;border-radius:4px;font-size:1.4rem;font-weight:600;background:#ef444425;color:#fca5a5;border:1px solid #ef444460}
.controls{display:flex;gap:8px;flex-shrink:0}
.controls input[type=file]{display:none}
.btn{padding:8px 14px;border:none;border-radius:6px;font-size:.85rem;font-weight:600;cursor:pointer}
.btn-primary{background:#f59e0b;color:#0a0e1a}.btn-primary:hover{background:#fbbf24}
.btn-secondary{background:#334155;color:#e2e8f0}.btn-secondary:hover{background:#475569}
.btn-danger{background:#ef4444;color:#fff}.btn-danger:hover{background:#f87171}
.rtsp-row{display:flex;gap:6px;flex:1}
.rtsp-row input{flex:1;padding:8px;background:#0f172a;border:1px solid #334155;border-radius:6px;color:#e2e8f0;font-size:.85rem}
.right{background:#111827;border-left:1px solid #1e293b;display:flex;flex-direction:column;min-height:0}
.panel{padding:14px;border-bottom:1px solid #1e293b;flex-shrink:0}
.panel h3{font-size:1.1rem;color:#f59e0b;margin-bottom:10px}
.alert-row{display:flex;gap:8px;align-items:center;margin-bottom:8px}
.alert-row label{font-size:.85rem;color:#94a3b8;white-space:nowrap}
.alert-row input[type=text]{flex:1;padding:8px;background:#0a0e1a;border:1px solid #334155;border-radius:6px;color:#e2e8f0;font-size:.85rem}
.alert-row input[type=number]{width:70px;padding:8px;background:#0a0e1a;border:1px solid #334155;border-radius:6px;color:#e2e8f0;font-size:.85rem;text-align:center}
.toggle{position:relative;display:inline-block;width:44px;height:24px}
.toggle input{opacity:0;width:0;height:0}
.slider{position:absolute;inset:0;background:#334155;border-radius:24px;cursor:pointer;transition:.3s}
.slider:before{content:"";position:absolute;width:18px;height:18px;left:3px;bottom:3px;background:#fff;border-radius:50%;transition:.3s}
input:checked+.slider{background:#f59e0b}
input:checked+.slider:before{transform:translateX(20px)}
.chat-feed{flex:1;overflow-y:auto;padding:14px;display:flex;flex-direction:column;gap:8px;min-height:0}
.msg{padding:10px 12px;border-radius:8px;font-size:.9rem;line-height:1.5}
.msg-user{background:#1e3a5f;align-self:flex-end;color:#93c5fd;max-width:90%}
.msg-ai{background:#1e293b;align-self:flex-start;color:#e2e8f0;max-width:95%}
.msg-alert{background:#ef444415;border:1px solid #ef4444;color:#fca5a5;align-self:stretch}
.msg-safe{background:#22c55e15;border:1px solid #22c55e;color:#4ade80;align-self:stretch}
.msg-warning{background:#eab30815;border:1px solid #eab308;color:#fde047;align-self:stretch}

</style></head><body>
<div class="header">
  <h1>🔍 SafetyLens v2</h1>
  <span>Continuous Warehouse Safety Monitor – powered by Qwen3-VL-30B on EKS Hybrid Nodes (NVIDIA DGX Spark)</span>
</div>
<div class="main">
  <div class="left">
    <div class="video-wrap" id="videoWrap">
      <img id="videoFrame" src="" alt="No feed — upload a video or connect RTSP">
      <div class="video-overlay" id="overlay"></div>
      <div class="alert-overlay" id="alertOverlay"><span class="alert-icon">⚠️</span></div>
      <div class="score-big" id="scoreBig" style="display:none"></div>
      <div class="hazard-tags" id="hazardTags"></div>
    </div>
    <div class="controls">
      <button class="btn btn-primary" onclick="document.getElementById('fileInput').click()">📁 Upload Video</button>
      <input type="file" id="fileInput" accept="video/*" onchange="uploadVideo(this)">
      <div class="rtsp-row">
        <input id="rtspUrl" placeholder="rtsp://... or /tmp/video.mp4 path">
        <button class="btn btn-secondary" onclick="startRtsp()">▶ Load</button>
      </div>
      <button class="btn btn-danger" onclick="stopFeed()">⏹ Stop</button>
    </div>
  </div>
  <div class="right">
    <div class="panel">
      <h3>🚨 Alert Configuration</h3>
      <div class="alert-row">
        <label>Prompt:</label>
        <input type="text" id="alertPrompt" value="Analyze for safety hazards." placeholder="Alert prompt...">
      </div>
      <div class="alert-row">
        <label>Score threshold ≤</label>
        <input type="number" id="alertThreshold" value="40" min="0" max="100">
        <label style="margin-left:12px">Enable alerts:</label>
        <label class="toggle"><input type="checkbox" id="alertToggle" onchange="toggleAlert()"><span class="slider"></span></label>
        <span id="alertStatus" style="font-size:.8rem;color:#94a3b8;margin-left:8px">Off</span>
      </div>
      <div class="alert-row">
        <label>Webhook URL:</label>
        <input type="text" id="webhookUrl" placeholder="https://your-endpoint/alert (optional)">
      </div>
    </div>
    <div class="chat-feed" id="chatFeed"></div>
  
  </div>
</div>
<script>
let ws = null;

function connect() {
  ws = new WebSocket('ws://' + location.host + '/ws');
  ws.onmessage = e => handleMsg(JSON.parse(e.data));
  ws.onclose = () => setTimeout(connect, 2000);
}

function handleMsg(msg) {
  if (msg.type === 'frame') {
    const vid = document.getElementById('videoFrame');
    if (!vid.src || vid.src === window.location.href) addMsg('Video playing', 'safe');
    vid.src = 'data:image/jpeg;base64,' + msg.b64;
  } else if (msg.type === 'vlm') {
    if (msg.is_safety || msg.is_alert) {
      const a = msg.analysis || {};
      updateScoreUI(a);
      if (msg.alert_triggered) {
        addMsg('ALERT: Safety score ' + msg.alert_score + '/100 — ' + (a.summary || ''), 'alert');
        playAlert();
        const ov = document.getElementById('alertOverlay');
        ov.style.display = 'flex';
        setTimeout(() => ov.style.display = 'none', 5000);
      } else if (msg.is_safety) {
        const score = a.score != null ? a.score : '?';
        const cls = score >= 70 ? 'safe' : score >= 40 ? 'warning' : 'alert';
        addMsg('Score: ' + score + '/100 — ' + (a.summary || ''), cls);
      }
    } else {
      addMsg(msg.raw, 'ai');
    }
  }
}

function updateScoreUI(a) {
  const score = a.score != null ? a.score : null;
  if (score === null) return;
  const big = document.getElementById('scoreBig');
  big.style.display = 'block';
  big.textContent = score + '/100';
  big.style.color = score >= 70 ? '#4ade80' : score >= 40 ? '#fde047' : '#fca5a5';
  document.getElementById('hazardTags').innerHTML = (a.deductions || []).map(d =>
    '<span class="tag">' + d.type + '</span>').join('');
  const level = score >= 70 ? ['badge-green','SAFE'] : score >= 40 ? ['badge-yellow','WARNING'] : ['badge-red','DANGER'];
  document.getElementById('overlay').innerHTML = '<span class="badge ' + level[0] + '">' + level[1] + '</span>';
}

function addMsg(text, type) {
  const feed = document.getElementById('chatFeed');
  const div = document.createElement('div');
  div.className = 'msg msg-' + type;
  div.textContent = text;
  feed.appendChild(div);
  feed.scrollTop = feed.scrollHeight;
}

async function uploadVideo(input) {
  const file = input.files[0]; if (!file) return;
  addMsg('Uploading ' + file.name + ' (' + (file.size/1024/1024).toFixed(1) + 'MB)...', 'ai');
  const fd = new FormData(); fd.append('video', file);
  try {
    const r = await fetch('/api/upload', {method:'POST', body:fd});
    if (!r.ok) { addMsg('Upload failed: HTTP ' + r.status, 'alert'); return; }
    addMsg('Processing ' + file.name + '... waiting for first frame', 'ai');
  } catch(e) { addMsg('Upload error: ' + e.message, 'alert'); }
  input.value = '';
}

async function startRtsp() {
  const url = document.getElementById('rtspUrl').value.trim();
  if (!url) { addMsg('Please enter a file path or RTSP URL', 'alert'); return; }
  addMsg('Processing ' + url + '... please wait', 'ai');
  try {
    const r = await fetch('/api/rtsp', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({url})});
    const data = await r.json();
    if (data.error) { addMsg('Error: ' + data.error, 'alert'); return; }
    addMsg('Video loaded — waiting for first frame', 'ai');
  } catch(e) { addMsg('Load error: ' + e.message, 'alert'); }
}

async function stopFeed() {
  await fetch('/api/stop', {method:'POST'});
  document.getElementById('videoFrame').src = '';
  document.getElementById('scoreBig').style.display = 'none';
  document.getElementById('hazardTags').innerHTML = '';
  document.getElementById('overlay').innerHTML = '';
  addMsg('Feed stopped', 'ai');
}

async function toggleAlert() {
  const active = document.getElementById('alertToggle').checked;
  document.getElementById('alertStatus').textContent = active ? 'Active' : 'Off';
  const prompt = document.getElementById('alertPrompt').value;
  const threshold = parseInt(document.getElementById('alertThreshold').value);
  await fetch('/api/alert', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({active, prompt, threshold, webhook_url: document.getElementById('webhookUrl').value.trim() || null})});
  addMsg(active ? 'Alert monitoring enabled — will notify when score <= ' + threshold : 'Alert monitoring disabled', 'ai');
}

async function sendQuery() {
  const input = document.getElementById('queryInput');
  const q = input.value.trim(); if (!q) return;
  input.value = '';
  addMsg(q, 'user');
  await fetch('/api/query', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({question:q})});
}

function playAlert() {
  try {
    const ctx = new AudioContext();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.connect(gain); gain.connect(ctx.destination);
    osc.frequency.value = 880;
    gain.gain.setValueAtTime(0.3, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.5);
    osc.start(); osc.stop(ctx.currentTime + 0.5);
  } catch(e) {}
}

connect();
</script></body></html>"""
