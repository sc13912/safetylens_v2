import asyncio, base64, cv2, httpx, io, json, logging, os, queue, time
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image
from threading import Thread, Lock
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from datetime import datetime, timezone
import boto3
from langfuse import Langfuse

LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_HOST = os.getenv("LANGFUSE_BASE_URL", "https://us.cloud.langfuse.com")
MEMORY_ID = os.getenv("MEMORY_ID", "")
langfuse = None
if LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY:
    langfuse = Langfuse(public_key=LANGFUSE_PUBLIC_KEY, secret_key=LANGFUSE_SECRET_KEY, host=LANGFUSE_HOST)
    logger.info("Langfuse observability enabled")
memory_client = None
if MEMORY_ID:
    memory_client = boto3.client("bedrock-agentcore", region_name=os.getenv("AWS_DEFAULT_REGION", "ap-southeast-2"))
    logger.info(f"AgentCore Memory enabled: {MEMORY_ID}")



VLLM_URL       = os.getenv("VLLM_URL",   "http://qwen3vl-30b-nvfp4:8000/v1")
MODEL_NAME     = os.getenv("MODEL_NAME", "Qwen3-VL-30B-A3B-Instruct-NVFP4")
FRAME_INTERVAL = float(os.getenv("FRAME_INTERVAL", "2.0"))

# --- DETAILED PROMPT (commented out - too relaxed scoring with group merging) ---
# SAFETY_PROMPT_DETAILED = """You are a warehouse safety auditor. Analyze this warehouse/factory image and evaluate safety hazards.
#
# Return ONLY valid JSON (no markdown, no explanation) in this exact format:
# {"score": <int 0-100>, "deductions": [{"type": "<hazard type>", "points": <negative int>, "detail": "<brief description>"}], "summary": "<one sentence overall assessment>"}
#
# Scoring: Start at 100 and deduct points for each hazard CATEGORY found.
# IMPORTANT RULES:
# - Each category is deducted AT MOST ONCE regardless of how many instances you see.
# - If you see 3 workers without hard hats, that is still only ONE PPE violation (-10 total, NOT -30).
# - Some categories are MERGED — only deduct for the group once:
#   GROUP A: "Blocked Exit / Fire Access" and "Fire Hazard" are ONE group → deduct max -30 once
#   GROUP B: "Stacking Safety" and "Unstable Load" are ONE group → deduct max -25 once
#   GROUP C: "Liquid Spill" and "Chemical Hazard" are ONE group → deduct max -35 once
#
# Hazard categories and max deduction per group:
# - PPE Violation: -10 (missing hard hat, vest, goggles, gloves)
# - Blocked Exit / Fire Hazard (Group A): -30 (obstructed exits, fire doors, flammable materials near heat, no extinguisher)
# - Stacking / Load Safety (Group B): -25 (overloaded shelves, leaning pallets, unstable stacks)
# - Trip Hazard: -20 (cables on floor, debris, uneven surfaces, items in walkways)
# - Forklift Hazard: -15 (forklift near pedestrians, no warning signs, unsafe operation)
# - Spill / Chemical Hazard (Group C): -35 (spills on floor, unmarked chemicals, no containment)
#
# If no hazards are visible, return score 100 with empty deductions.
# Only report hazards you can clearly see. Do NOT guess."""
# --- END DETAILED PROMPT ---

SAFETY_PROMPT = """Analyze this warehouse/factory image for safety hazards.
Return ONLY compact single-line JSON (no extra whitespace or newlines):
{"deductions":[{"type":"<exact category name>"}],"summary":"<max 12 words>"}
Categories (use these EXACT names, each at most once):
PPE violation, Forklift Hazard, Trip Hazard, Stacking/Load Safety, Blocked Fire/Emergency Exit, Spill/Chemical Hazard, Fire/Electrical Hazard
Only report clearly visible hazards. Use an empty deductions list if safe."""

# ── State ──────────────────────────────────────────────────────────────────────
state = {
    "cap": None, "running": False, "loop_video": True,
    "current_frame_b64": None, "last_analysis": None,
    "alert_prompt": None, "alert_threshold": 40, "alert_active": False, "webhook_url": None,
    "vlm_busy": False, "frame_seq": 0,
}
state_lock = Lock()
ws_clients: list[WebSocket] = []
broadcast_queue: queue.Queue = queue.Queue()
memory_queue: queue.Queue = queue.Queue()
cameras = {}  # {camera_id: {"b64": str, "mime": str, "analysis": dict}}


# ── VLM helpers ────────────────────────────────────────────────────────────────
def encode_frame(frame: np.ndarray) -> str | None:
    h, w = frame.shape[:2]
    if h < 28 or w < 28:
        return None
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)  # native resolution: send the frame untouched (no resize)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
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


# ── Deterministic, model-independent safety scoring ─────────────────────────────
# The VLM reliably IDENTIFIES + LABELS hazards, but its self-reported score/points
# drift between models and it can't do the arithmetic in no-think mode. So we IGNORE
# the model's numbers and score from the detected hazard TYPES using a fixed rubric,
# each category counted at most once, PLUS severity banding:
#   base = max(0, 100 - sum(category points))
#   MAJOR hazard present (Spill/Chemical or Fire/Electrical) -> DANGER (cap <= 35)
#   moderate-only hazards                                    -> floor at 60 (never DANGER)
# NOTE: "Blocked Fire/Emergency Exit" is moderate (-25) and is matched BEFORE
# "Fire/Electrical Hazard" so a blocked fire exit is NOT treated as an active fire.
HAZARD_RUBRIC = [
    # (canonical label, points, keywords, is_major)
    ("Spill/Chemical Hazard", -50, ("spill", "chemical"), True),
    ("Blocked Fire/Emergency Exit", -35, ("blocked", "emergency", "exit", "egress"), False),
    ("Fire/Electrical Hazard", -65, ("fire", "electrical", "flammable", "wiring", "spark", "combustible"), True),
    ("Stacking/Load Safety", -25, ("stack", "load", "unstable"), False),
    ("Trip Hazard", -20, ("trip",), False),
    ("Forklift Hazard", -15, ("forklift",), False),
    ("PPE violation", -10, ("ppe", "hard hat", "hardhat", "helmet", "vest", "goggle", "glove"), False),
]
DANGER_CAP = 35          # any MAJOR hazard -> score forced to DANGER band (<40)
MODERATE_FLOOR = 60      # moderate-only hazards never fall below WARNING


def _categorize_hazard(hazard_type: str):
    t = (hazard_type or "").lower()
    for label, pts, kws, major in HAZARD_RUBRIC:
        if any(k in t for k in kws):
            return label, pts, major
    return None, None, None


def compute_safety_score(deductions):
    """Recompute score from detected hazard types (each category once) + severity banding."""
    seen = {}
    if isinstance(deductions, dict):
        deductions = [deductions]
    if not isinstance(deductions, list):
        deductions = []
    for d in deductions:
        if isinstance(d, str):
            type_str, detail, mp = d, "", 0
        elif isinstance(d, dict):
            type_str, detail = d.get("type", ""), d.get("detail", "")
            try:
                mp = int(d.get("points", 0) or 0)
            except Exception:
                mp = 0
        else:
            continue
        label, pts, major = _categorize_hazard(type_str)
        if label is None:
            label, pts, major = (type_str or "Other Hazard"), -abs(mp), False
        if label not in seen:
            seen[label] = {"type": label, "points": pts, "detail": detail, "_major": major}
    norm = list(seen.values())
    base = max(0, 100 + sum(x["points"] for x in norm))
    has_major = any(x["_major"] for x in norm)
    if has_major:
        score = min(base, DANGER_CAP)   # C: additive, majors forced into DANGER (ceiling)
    else:
        score = base                    # pure additive
    for x in norm:
        x.pop("_major", None)
    return max(0, min(100, score)), norm


def parse_safety_json(raw: str) -> dict:
    raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
    try:
        data = json.loads(raw)
    except Exception:
        return {"score": -1, "deductions": [], "summary": raw[:200]}
    score, norm = compute_safety_score(data.get("deductions"))
    data["deductions"] = norm
    data["score"] = score
    data.setdefault("summary", "")
    return data


async def query_cameras(question: str) -> str:
    if not cameras:
        return "No camera feeds loaded yet."
    cam_summaries = []
    for cid, cam in cameras.items():
        a = cam.get("analysis", {})
        score = a.get("score", "N/A")
        level = "🟢 GREEN" if score >= 70 else "🟡 YELLOW" if score >= 40 else "🔴 RED"
        deds = ", ".join(d["type"] for d in a.get("deductions", [])) or "None"
        cam_summaries.append(f"Camera {cid}: Score {score}/100 ({level}), Hazards: {deds}, Summary: {a.get('summary', 'N/A')}")
    # Include video stream analysis if available
    with state_lock:
        va = state.get("last_analysis")
    if va:
        vs = va.get("score", "N/A")
        vl = "🟢 GREEN" if vs >= 70 else "🟡 YELLOW" if vs >= 40 else "🔴 RED"
        vd = ", ".join(d["type"] for d in va.get("deductions", [])) or "None"
        cam_summaries.append(f"Video Stream: Score {vs}/100 ({vl}), Hazards: {vd}, Summary: {va.get('summary', 'N/A')}")
    context = "\n".join(cam_summaries)
    messages = [
        {"role": "system", "content": f"You are a warehouse safety assistant. Here are the current analyses:\n\n{context}\n\nAnswer the user question based on this data. ALWAYS format responses with: emoji status indicators (🔴🟡🟢✅❌⚠️), markdown tables for comparisons, bullet points for lists, and bold for key values. Be concise but visually rich."},
        {"role": "user", "content": question},
    ]
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(f"{VLLM_URL}/chat/completions", json={
            "model": MODEL_NAME, "messages": messages, "max_tokens": 512, "temperature": 0.3,
        })
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


# ── Frame loop ─────────────────────────────────────────────────────────────────
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
            state["frame_seq"] += 1
        now = time.time()
        with state_lock:
            busy = state["vlm_busy"]
        if not busy and (now - last_vlm_time) >= FRAME_INTERVAL:
            with state_lock:
                state["vlm_busy"] = True
                threshold = state["alert_threshold"]
            prompt = SAFETY_PROMPT
            vlm_b64 = encode_frame(frame)
            if vlm_b64 is None:
                with state_lock:
                    state["vlm_busy"] = False
                last_vlm_time = now
                continue
            def run_vlm(b=vlm_b64, thr=threshold):
                raw = call_vlm_sync(SAFETY_PROMPT, b)
                parsed = parse_safety_json(raw)
                result = {"type": "vlm", "is_safety": True, "analysis": parsed, "raw": raw, "prompt": SAFETY_PROMPT}
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
                            httpx.post(webhook, json={"score": score, "hazards": [d["type"] for d in parsed.get("deductions", [])], "summary": parsed.get("summary", "")}, timeout=5.0)
                        except Exception as _e:
                            logger.warning(f"Webhook failed: {_e}")
                # Report to Langfuse + AgentCore Memory
                score_val = parsed.get("score", -1)
                if score_val == -1 and langfuse:
                    try:
                        obs = langfuse.start_observation(name="vlm-parse-error", as_type="generation", model=MODEL_NAME, input=SAFETY_PROMPT, output=raw, level="ERROR")
                        obs.end()
                    except Exception:
                        pass
                if score_val != -1:
                    if langfuse:
                        try:
                            obs = langfuse.start_observation(name="safety-analysis", as_type="generation", model=MODEL_NAME, input=SAFETY_PROMPT, output=raw)
                            obs.end()
                            langfuse.create_score(name="safety-score", value=score_val / 100, trace_id=obs.trace_id)
                        except Exception as e:
                            logger.warning(f"Langfuse report failed: {e}")
                    if memory_client and MEMORY_ID:
                        try:
                            event_ts = datetime.now(timezone.utc)
                            ts = event_ts.isoformat()
                            hazards = ", ".join(d["type"] for d in parsed.get("deductions", []))
                            text = f"[{ts}] Safety score: {score_val}/100. Hazards: {hazards or 'None'}. Summary: {parsed.get('summary', '')}"
                            memory_queue.put({"event_ts": event_ts, "text": text})
                        except Exception as e:
                            logger.warning(f"AgentCore Memory write failed: {e}")
                with state_lock:
                    state["vlm_busy"] = False
                broadcast_queue.put(result)
            Thread(target=run_vlm, daemon=True).start()
            last_vlm_time = now
        time.sleep(1 / 30)


def memory_worker():
    # Drains Memory writes off the analysis critical path. eventTimestamp is captured
    # at analysis time and passed through, so the recorded time is exact regardless
    # of when the (slower) AgentCore write actually lands.
    while True:
        job = memory_queue.get()
        try:
            if memory_client and MEMORY_ID:
                memory_client.create_event(
                    eventTimestamp=job["event_ts"],
                    memoryId=MEMORY_ID,
                    actorId="safetylens-vlm",
                    sessionId="cctv-monitoring",
                    payload=[{"conversational": {"role": "ASSISTANT", "content": {"text": job["text"]}}}],
                )
        except Exception as e:
            logger.warning(f"AgentCore Memory write failed: {e}")


async def broadcast_pump():
    last_seq = -1
    while True:
        # Deliver ALL pending events (vlm results, stopped) reliably.
        while True:
            try:
                msg = broadcast_queue.get_nowait()
            except queue.Empty:
                break
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
        # Send ONLY the latest frame (drop intermediates -> no unbounded backlog).
        with state_lock:
            seq = state["frame_seq"]
            fb = state["current_frame_b64"]
        if fb is not None and seq != last_seq:
            last_seq = seq
            fmsg = {"type": "frame", "b64": fb}
            dead = []
            for ws in ws_clients:
                try:
                    await ws.send_json(fmsg)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                ws_clients.remove(ws)
        await asyncio.sleep(0.033)


@asynccontextmanager
async def lifespan(app: FastAPI):
    Thread(target=frame_loop, daemon=True).start()
    Thread(target=memory_worker, daemon=True).start()
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
        state["loop_video"] = True
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
        state["current_frame_b64"] = None
        state["last_analysis"] = None
        if state["cap"]:
            state["cap"].release()
            state["cap"] = None
    # Drain the broadcast queue to stop stale frames
    while not broadcast_queue.empty():
        try:
            broadcast_queue.get_nowait()
        except queue.Empty:
            break
    # Send a stop signal to all WebSocket clients
    broadcast_queue.put({"type": "stopped"})
    return {"status": "stopped"}


@app.post("/api/alert")
async def set_alert(request: Request):
    body = await request.json()
    active = body.get("active", False)
    with state_lock:
        state["alert_active"] = active
        state["alert_prompt"] = SAFETY_PROMPT if active else None
        state["alert_threshold"] = int(body.get("threshold", 40))
        state["webhook_url"] = body.get("webhook_url") or None
    return {"status": "ok"}


@app.post("/api/cam/upload/{camera_id}")
async def cam_upload(camera_id: int, image: UploadFile = File(...)):
    data = await image.read()
    b64 = base64.b64encode(data).decode()
    mime = image.content_type or "image/jpeg"
    raw = call_vlm_sync(SAFETY_PROMPT, b64)
    analysis = parse_safety_json(raw)
    cameras[camera_id] = {"b64": b64, "mime": mime, "analysis": analysis}
    return {"camera_id": camera_id, "analysis": analysis}


@app.get("/api/cameras")
async def get_cameras():
    return {cid: {"analysis": cam["analysis"], "has_image": True} for cid, cam in cameras.items()}


@app.post("/api/query")
async def query(request: Request):
    body = await request.json()
    answer = await query_cameras(body.get("question", ""))
    return {"answer": answer}


@app.get("/api/status")
async def status():
    with state_lock:
        return {"running": state["running"], "last_analysis": state["last_analysis"], "alert_active": state["alert_active"]}


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
        await ws.send_json({"type": "vlm", "is_safety": True, "analysis": analysis, "raw": analysis.get("summary", ""), "prompt": SAFETY_PROMPT})
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
.header span{color:#94a3b8;font-size:1.2rem}
.main{display:grid;grid-template-columns:1fr 308px 430px;flex:1;min-height:0}
/* Left: video */
.left{display:flex;flex-direction:column;padding:12px;gap:10px;min-height:0}
.video-wrap{position:relative;background:#000;border-radius:10px;overflow:hidden;flex:1;min-height:0}
.video-wrap img{width:90%;height:90%;object-fit:contain;margin:auto;position:absolute;top:5%;left:5%}
.video-overlay{position:absolute;top:8px;left:8px;display:flex;gap:6px;flex-wrap:wrap}
.alert-overlay{display:none;position:absolute;inset:0;background:#ef444440;align-items:center;justify-content:center;z-index:10;animation:pulse 0.5s infinite}
.alert-overlay .alert-icon{font-size:8rem;filter:drop-shadow(0 0 20px #ef4444)}
.badge{padding:6px 14px;border-radius:6px;font-size:2rem;font-weight:900;text-shadow:0 2px 8px #000}
.badge-green{background:#22c55e30;color:#4ade80;border:1px solid #22c55e}
.badge-yellow{background:#eab30830;color:#fde047;border:1px solid #eab308}
.badge-red{background:#ef444430;color:#fca5a5;border:1px solid #ef4444;animation:pulse 1s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.6}}
.score-big{position:absolute;top:8px;right:8px;font-size:2rem;font-weight:900;text-shadow:0 2px 8px #000}
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
/* Middle: camera feeds */
.middle{background:#0f172a;border-left:1px solid #1e293b;border-right:1px solid #1e293b;display:flex;flex-direction:column;gap:8px;padding:8px;min-height:0;overflow:auto}
.cam{border-radius:8px;border:3px solid #334155;background:#111827;display:flex;flex-direction:column;overflow:hidden;flex:1;min-height:0;transition:border-color .3s}
.cam.green{border-color:#22c55e;box-shadow:0 0 10px #22c55e30}
.cam.yellow{border-color:#eab308;box-shadow:0 0 10px #eab30830}
.cam.red{border-color:#ef4444;box-shadow:0 0 10px #ef444430}
.cam-header{padding:6px 10px;display:flex;justify-content:space-between;align-items:center;background:#0f172a;font-size:1.02rem;font-weight:600;flex-shrink:0}
.cam-header .id{color:#94a3b8}.cam-header .score{font-weight:700;font-size:1.08rem}
.score-green{color:#22c55e}.score-yellow{color:#eab308}.score-red{color:#ef4444}
.cam-body{flex:1;display:flex;align-items:center;justify-content:center;cursor:pointer;position:relative;min-height:0;overflow:hidden}
.cam-body img{width:100%;height:100%;object-fit:contain;position:absolute;top:0;left:0}
.cam-body .placeholder{color:#475569;font-size:.75rem;text-align:center;padding:8px}
.cam-footer{padding:4px 8px;font-size:.7rem;color:#94a3b8;background:#0f172a;flex-shrink:0}
.cam-footer .hazards{display:flex;flex-wrap:wrap;gap:3px;margin-top:2px}
.cam-tag{padding:2px 6px;border-radius:3px;font-size:.65rem;font-weight:600;background:#ef444425;color:#fca5a5}
.cam-tag-safe{background:#3b82f625;color:#93c5fd}
.cam-loading{position:absolute;inset:0;background:#111827ee;display:flex;align-items:center;justify-content:center;z-index:2}
.cam-loading .spinner-lg{width:24px;height:24px;border:3px solid #334155;border-top-color:#f59e0b;border-radius:50%;animation:spin .8s linear infinite}
input[type=file]{display:none}
/* Right: alerts + query */
.right{background:#111827;display:flex;flex-direction:column;min-height:0}
.panel{padding:12px;border-bottom:1px solid #1e293b;flex-shrink:0}
.panel h3{font-size:1.1rem;color:#f59e0b;margin-bottom:8px}
.alert-row{display:flex;gap:8px;align-items:center;margin-bottom:6px}
.alert-row label{font-size:.8rem;color:#94a3b8;white-space:nowrap}
.alert-row input[type=text]{flex:1;padding:6px;background:#0a0e1a;border:1px solid #334155;border-radius:6px;color:#e2e8f0;font-size:.8rem}
.alert-row input[type=number]{width:60px;padding:6px;background:#0a0e1a;border:1px solid #334155;border-radius:6px;color:#e2e8f0;font-size:.8rem;text-align:center}
.toggle{position:relative;display:inline-block;width:40px;height:22px}
.toggle input{opacity:0;width:0;height:0}
.slider{position:absolute;inset:0;background:#334155;border-radius:22px;cursor:pointer;transition:.3s}
.slider:before{content:"";position:absolute;width:16px;height:16px;left:3px;bottom:3px;background:#fff;border-radius:50%;transition:.3s}
input:checked+.slider{background:#f59e0b}
input:checked+.slider:before{transform:translateX(18px)}
.chat-feed{flex:1;overflow-y:auto;padding:10px;display:flex;flex-direction:column;gap:6px;min-height:0}
.msg{padding:8px 10px;border-radius:6px;font-size:.85rem;line-height:1.4}
.msg-user{background:#1e3a5f;align-self:flex-end;color:#93c5fd;max-width:90%}
.msg-ai{background:#1e293b;align-self:flex-start;color:#e2e8f0;max-width:95%}
.msg-ai table{width:100%;border-collapse:collapse;margin:8px 0;font-size:.8rem}
.msg-ai th,.msg-ai td{border:1px solid #334155;padding:4px 8px;text-align:left}
.msg-ai th{background:#0f172a;color:#f59e0b}
.msg-ai ul,.msg-ai ol{padding-left:18px;margin:4px 0}
.msg-ai code{background:#0a0e1a;padding:1px 4px;border-radius:3px;font-size:.8rem}
.msg-alert{background:#ef444415;border:1px solid #ef4444;color:#fca5a5;align-self:stretch}
.msg-safe{background:#22c55e15;border:1px solid #22c55e;color:#4ade80;align-self:stretch}
.msg-warning{background:#eab30815;border:1px solid #eab308;color:#fde047;align-self:stretch}
.query-box{padding:10px;border-top:1px solid #1e293b;display:flex;gap:6px;flex-shrink:0}
.drag-handle{height:6px;background:#1e293b;cursor:ns-resize;flex-shrink:0;display:flex;align-items:center;justify-content:center}
.drag-handle:hover{background:#334155}
.drag-handle::after{content:'';width:40px;height:2px;background:#475569;border-radius:1px}
.query-box input{flex:1;padding:8px;background:#0a0e1a;border:1px solid #334155;border-radius:6px;color:#e2e8f0;font-size:.85rem}
.query-box button{padding:8px 12px;background:#f59e0b;color:#0a0e1a;border:none;border-radius:6px;font-weight:700;cursor:pointer;font-size:.85rem}
.spinner-sm{display:inline-block;width:12px;height:12px;border:2px solid #475569;border-top-color:#f59e0b;border-radius:50%;animation:spin .6s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
</style></head><body>
<div class="header">
  <h1>🔍 SafetyLens v2</h1>
  <span>Warehouse Safety Visual Alerts Agent – powered by Qwen3-VL deployed on Amazon EKS Hybrid Nodes (NVIDIA DGX Spark)</span>
</div>
<div class="main">
  <!-- LEFT: Video Stream -->
  <div class="left">
    <div class="video-wrap">
      <img id="videoFrame" src="" alt="No feed">
      <div class="video-overlay" id="overlay"></div>
      <div class="alert-overlay" id="alertOverlay"><span class="alert-icon">⚠️</span></div>
      <div class="score-big" id="scoreBig" style="display:none"></div>
      <div class="hazard-tags" id="hazardTags"></div>
    </div>
    <div class="controls">
      <button class="btn btn-primary" onclick="document.getElementById('fileInput').click()">📁 Upload Video</button>
      <input type="file" id="fileInput" accept="video/*" onchange="uploadVideo(this)">
      <div class="rtsp-row">
        <input id="rtspUrl" placeholder="/tmp/video.mp4 or rtsp://...">
        <button class="btn btn-secondary" onclick="startRtsp()">▶ Load</button>
      </div>
      <button class="btn btn-danger" onclick="stopFeed()">⏹ Stop</button>
    </div>
  </div>
  <!-- MIDDLE: Camera Feeds -->
  <div class="middle">
    <div class="cam" id="cam1">
      <div class="cam-header"><span class="id">Camera #1</span><span class="score" id="cscore1">—</span></div>
      <div class="cam-body" id="cbody1" onclick="document.getElementById('cfile1').click()">
        <div class="placeholder" id="cph1">📷 Click to upload</div>
        <img id="cimg1" style="display:none">
      </div>
      <div class="cam-footer" id="cfooter1">No feed</div>
      <input type="file" id="cfile1" accept="image/*" onchange="uploadCam(1,this)">
    </div>
    <div class="cam" id="cam2">
      <div class="cam-header"><span class="id">Camera #2</span><span class="score" id="cscore2">—</span></div>
      <div class="cam-body" id="cbody2" onclick="document.getElementById('cfile2').click()">
        <div class="placeholder" id="cph2">📷 Click to upload</div>
        <img id="cimg2" style="display:none">
      </div>
      <div class="cam-footer" id="cfooter2">No feed</div>
      <input type="file" id="cfile2" accept="image/*" onchange="uploadCam(2,this)">
    </div>
    <div class="cam" id="cam3">
      <div class="cam-header"><span class="id">Camera #3</span><span class="score" id="cscore3">—</span></div>
      <div class="cam-body" id="cbody3" onclick="document.getElementById('cfile3').click()">
        <div class="placeholder" id="cph3">📷 Click to upload</div>
        <img id="cimg3" style="display:none">
      </div>
      <div class="cam-footer" id="cfooter3">No feed</div>
      <input type="file" id="cfile3" accept="image/*" onchange="uploadCam(3,this)">
    </div>
  </div>
  <!-- RIGHT: Alerts + Analysis Feed + Query -->
  <div class="right">
    <div class="panel">
      <h3>🚨 Alert Configuration</h3>
      <div class="alert-row">
        <label>Score threshold ≤</label>
        <input type="number" id="alertThreshold" value="40" min="0" max="100">
        <label style="margin-left:8px">Enable alerts:</label>
        <label class="toggle"><input type="checkbox" id="alertToggle" onchange="toggleAlert()"><span class="slider"></span></label>

      </div>
      <div class="alert-row">
        <label>Webhook URL:</label>
        <input type="text" id="webhookUrl" placeholder="https://... (optional)">
      </div>
    </div>
    <div class="chat-feed" id="analysisFeed" style="flex:1;min-height:0"></div>
    <div class="drag-handle" id="dragHandle"></div>
    <div class="panel" style="flex-shrink:0;border-bottom:none">
      <h3>💬 Safety Query</h3>
    </div>
    <div class="chat-feed" id="queryFeed" style="flex:1;min-height:0"></div>
    <div class="query-box">
      <input id="queryInput" placeholder="Ask about cameras or video..." onkeydown="if(event.key==='Enter')sendQuery()">
      <button onclick="sendQuery()">Ask</button>
    </div>
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
  if (msg.type === 'stopped') {
    document.getElementById('videoFrame').src = '';
    document.getElementById('scoreBig').style.display = 'none';
    document.getElementById('hazardTags').innerHTML = '';
    document.getElementById('overlay').innerHTML = '';
    return;
  }
  if (msg.type === 'frame') {
    const vid = document.getElementById('videoFrame');
    if (!vid.src || vid.src === window.location.href) addMsg('Video playing', 'safe', 'analysisFeed');
    vid.src = 'data:image/jpeg;base64,' + msg.b64;
  } else if (msg.type === 'vlm') {
    if (msg.is_safety) {
      const a = msg.analysis || {};
      updateVideoScore(a);
      const score = a.score != null ? a.score : '?';
      const cls = score >= 70 ? 'safe' : score >= 40 ? 'warning' : 'alert';
      addMsg('Score: ' + score + '/100 — ' + (a.summary || ''), cls, 'analysisFeed');
      if (msg.alert_triggered) {
        addMsg('ALERT: Score ' + msg.alert_score + '/100 — ' + (a.summary || ''), 'alert', 'analysisFeed');
        playAlert();
        const ov = document.getElementById('alertOverlay');
        ov.style.display = 'flex';
        setTimeout(() => ov.style.display = 'none', 5000);
      }
    }
  }
}
function updateVideoScore(a) {
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
function addMsg(text, type, feedId) {
  feedId = feedId || 'analysisFeed';
  const feed = document.getElementById(feedId);
  const div = document.createElement('div');
  div.className = 'msg msg-' + type;
  div.textContent = text;
  feed.appendChild(div);
  feed.scrollTop = feed.scrollHeight;
}
function addHtmlMsg(html, type, feedId) {
  feedId = feedId || 'queryFeed';
  const feed = document.getElementById(feedId);
  const div = document.createElement('div');
  div.className = 'msg msg-' + type;
  div.innerHTML = html;
  feed.appendChild(div);
  feed.scrollTop = feed.scrollHeight;
}
function renderMd(t){
  // tables
  t=t.replace(/^(\|.+\|)\n(\|[-| :]+\|)\n((?:\|.+\|\n?)*)/gm,(_,hdr,_sep,body)=>{
    const th=hdr.split('|').filter(c=>c.trim()).map(c=>'<th>'+c.trim()+'</th>').join('');
    const rows=body.trim().split('\n').map(r=>'<tr>'+r.split('|').filter(c=>c.trim()).map(c=>'<td>'+c.trim()+'</td>').join('')+'</tr>').join('');
    return '<table><tr>'+th+'</tr>'+rows+'</table>';
  });
  // bold
  t=t.replace(/\*\*(.+?)\*\*/g,'<strong>$1</strong>');
  // bullets
  t=t.replace(/^[-*] (.+)$/gm,'<li>$1</li>');
  t=t.replace(/((?:<li>.*<\/li>\n?)+)/g,'<ul>$1</ul>');
  // numbered lists
  t=t.replace(/^\d+\. (.+)$/gm,'<li>$1</li>');
  // inline code
  t=t.replace(/`([^`]+)`/g,'<code>$1</code>');
  // line breaks
  t=t.replace(/\n/g,'<br>');
  return t;
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
  if (!url) { addMsg('Enter a file path or RTSP URL', 'alert'); return; }
  addMsg('Processing ' + url + '...', 'ai');
  try {
    const r = await fetch('/api/rtsp', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({url})});
    const data = await r.json();
    if (data.error) { addMsg('Error: ' + data.error, 'alert'); return; }
    addMsg('Video loaded', 'ai');
  } catch(e) { addMsg('Load error: ' + e.message, 'alert'); }
}
async function stopFeed() {
  // Clear UI immediately
  document.getElementById('videoFrame').src = '';
  document.getElementById('scoreBig').style.display = 'none';
  document.getElementById('hazardTags').innerHTML = '';
  document.getElementById('overlay').innerHTML = '';
  addMsg('Feed stopped', 'ai');
  // Then tell server
  fetch('/api/stop', {method:'POST'});
}
async function toggleAlert() {
  const active = document.getElementById('alertToggle').checked;
  
  const threshold = parseInt(document.getElementById('alertThreshold').value);
  await fetch('/api/alert', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({active, threshold, webhook_url: document.getElementById('webhookUrl').value.trim() || null})});
  addMsg(active ? 'Alert monitoring enabled — will notify when score <= ' + threshold : 'Alert monitoring disabled', 'ai');
}
async function uploadCam(id, input) {
  const file = input.files[0]; if (!file) return;
  const reader = new FileReader();
  reader.onload = e => { document.getElementById('cimg'+id).src = e.target.result; document.getElementById('cimg'+id).style.display='block'; document.getElementById('cph'+id).style.display='none'; };
  reader.readAsDataURL(file);
  const body = document.getElementById('cbody'+id);
  const loader = document.createElement('div'); loader.className='cam-loading'; loader.id='cloader'+id;
  loader.innerHTML='<div class="spinner-lg"></div>'; body.appendChild(loader);
  document.getElementById('cfooter'+id).textContent = 'Analyzing...';
  const fd = new FormData(); fd.append('image', file);
  try {
    const r = await fetch('/api/cam/upload/' + id, {method:'POST', body:fd});
    const data = await r.json();
    updateCam(id, data.analysis);
  } catch(e) { document.getElementById('cfooter'+id).textContent = 'Error'; }
  const l = document.getElementById('cloader'+id); if(l) l.remove();
  input.value = '';
}
function updateCam(id, a) {
  if (!a) return;
  const s = a.score;
  document.getElementById('cam'+id).className = 'cam ' + (s>=70?'green':s>=40?'yellow':'red');
  const se = document.getElementById('cscore'+id);
  se.textContent = s + '/100';
  se.className = 'score ' + (s>=70?'score-green':s>=40?'score-yellow':'score-red');
  const footer = document.getElementById('cfooter'+id);
  if (a.deductions && a.deductions.length) {
    footer.innerHTML = '<div class="hazards">' + a.deductions.map(d => '<span class="cam-tag">' + d.type + '</span>').join('') + '</div>';
  } else {
    footer.innerHTML = '<span class="cam-tag cam-tag-safe">✓ No hazards</span>';
  }
}
async function sendQuery() {
  const input = document.getElementById('queryInput');
  const q = input.value.trim(); if (!q) return;
  input.value = '';
  addMsg(q, 'user', 'queryFeed');
  addMsg('Thinking...', 'ai', 'queryFeed');
  try {
    const r = await fetch('/api/query', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({question:q})});
    const data = await r.json();
    const msgs = document.getElementById('queryFeed').querySelectorAll('.msg-ai');
    msgs[msgs.length-1].innerHTML = renderMd(data.answer);
  } catch(e) { addMsg('Error: ' + e.message, 'alert', 'queryFeed'); }
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
// Draggable divider
(function(){
  const handle = document.getElementById('dragHandle');
  const upper = document.getElementById('analysisFeed');
  const right = handle.closest('.right');
  let dragging = false, startY, startH;
  handle.addEventListener('mousedown', e => {
    dragging = true; startY = e.clientY; startH = upper.offsetHeight;
    document.body.style.cursor = 'ns-resize'; document.body.style.userSelect = 'none';
  });
  document.addEventListener('mousemove', e => {
    if (!dragging) return;
    const delta = e.clientY - startY;
    const newH = Math.max(60, Math.min(startH + delta, right.offsetHeight - 200));
    upper.style.flex = 'none'; upper.style.height = newH + 'px';
  });
  document.addEventListener('mouseup', () => {
    if (dragging) { dragging = false; document.body.style.cursor = ''; document.body.style.userSelect = ''; }
  });
})();
connect();
</script></body></html>"""
