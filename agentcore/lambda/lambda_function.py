import json, boto3, os

AGENT_ARN = os.environ["AGENT_RUNTIME_ARN"]
REGION = os.environ.get("AWS_REGION", "ap-southeast-2")
client = boto3.client("bedrock-agentcore", region_name=REGION)

HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SafetyLens Insights</title>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0a0e1a;color:#e2e8f0;height:100vh;display:flex;flex-direction:column}
.header{background:linear-gradient(135deg,#0f172a,#1e293b);padding:16px 24px;border-bottom:1px solid #334155;display:flex;align-items:center;gap:12px}
.header h1{font-size:1.3rem;color:#f59e0b}
.header span{color:#94a3b8;font-size:.9rem}
.chat{flex:1;overflow-y:auto;padding:20px;display:flex;flex-direction:column;gap:12px}
.msg{padding:12px 16px;border-radius:10px;font-size:.95rem;line-height:1.6;max-width:85%}
.msg-user{background:#1e3a5f;align-self:flex-end;color:#93c5fd;white-space:pre-wrap}
.msg-ai{background:#1e293b;align-self:flex-start;color:#e2e8f0}
.msg-ai h1,.msg-ai h2,.msg-ai h3{color:#f59e0b;margin:8px 0 4px}
.msg-ai table{width:100%;border-collapse:collapse;margin:8px 0;font-size:.85rem}
.msg-ai th,.msg-ai td{border:1px solid #334155;padding:4px 8px;text-align:left}
.msg-ai th{background:#0f172a;color:#f59e0b}
.msg-ai ul,.msg-ai ol{padding-left:18px;margin:4px 0}
.msg-ai code{background:#0a0e1a;padding:1px 4px;border-radius:3px;font-size:.85rem}
.msg-ai hr{border:none;border-top:1px solid #334155;margin:10px 0}
.msg-ai strong{color:#fbbf24}
.msg-system{background:#22c55e15;border:1px solid #22c55e40;color:#4ade80;align-self:center;font-size:.85rem}
.input-bar{padding:16px 20px;border-top:1px solid #334155;display:flex;gap:10px;background:#111827}
.input-bar input{flex:1;padding:12px 16px;background:#0a0e1a;border:1px solid #334155;border-radius:8px;color:#e2e8f0;font-size:.95rem}
.input-bar button{padding:12px 20px;background:#f59e0b;color:#0a0e1a;border:none;border-radius:8px;font-weight:700;cursor:pointer;font-size:.95rem}
.input-bar button:hover{background:#fbbf24}
.input-bar button:disabled{background:#334155;color:#64748b;cursor:not-allowed}
</style></head><body>
<div class="header">
<h1>SafetyLens Insights</h1>
<span>AI Safety Analyst - Ask about warehouse safety incidents</span>
</div>
<div class="chat" id="chat">
<div class="msg msg-system">Welcome! Ask me about safety incidents, hazard trends, or compliance questions.</div>
</div>
<div class="input-bar">
<input id="input" placeholder="e.g. What PPE violations happened today?" onkeydown="if(event.key==='Enter'&&!event.shiftKey)send()">
<button id="btn" onclick="send()">Ask</button>
</div>
<script>
async function send(){
  const input=document.getElementById('input'),btn=document.getElementById('btn');
  const q=input.value.trim();if(!q)return;
  input.value='';btn.disabled=true;
  addMsg(q,'user');
  addMsg('Thinking...','ai','thinking');
  try{
    const r=await fetch('/prod/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({prompt:q})});
    const data=await r.json();
    document.getElementById('thinking').remove();
    addMd(data.response||data.error||'No response');
  }catch(e){document.getElementById('thinking').remove();addMsg('Error: '+e.message,'ai');}
  btn.disabled=false;input.focus();
}
function addMsg(text,role,id){
  const chat=document.getElementById('chat');
  const div=document.createElement('div');
  div.className='msg msg-'+role;
  div.textContent=text;
  if(id)div.id=id;
  chat.appendChild(div);
  chat.scrollTop=chat.scrollHeight;
}
function addMd(md){
  const chat=document.getElementById('chat');
  const div=document.createElement('div');
  div.className='msg msg-ai';
  div.innerHTML=marked.parse(md);
  chat.appendChild(div);
  chat.scrollTop=chat.scrollHeight;
}
</script></body></html>"""


def lambda_handler(event, context):
    path = event.get("rawPath", event.get("path", "/"))
    method = event.get("requestContext", {}).get("http", {}).get("method",
             event.get("requestContext", {}).get("httpMethod",
             event.get("httpMethod", "GET")))

    if method == "GET" and path in ("/", "/prod", "/prod/"):
        return {"statusCode": 200, "headers": {"Content-Type": "text/html"}, "body": HTML}

    if method == "POST" and "/ask" in path:
        body = json.loads(event.get("body", "{}"))
        prompt = body.get("prompt", "")
        try:
            response = client.invoke_agent_runtime(
                agentRuntimeArn=AGENT_ARN,
                payload=json.dumps({"prompt": prompt}).encode()
            )
            result = response["response"].read().decode()
            if result.startswith('"') and result.endswith('"'):
                result = json.loads(result)
            return {"statusCode": 200, "headers": {"Content-Type": "application/json"}, "body": json.dumps({"response": result})}
        except Exception as e:
            return {"statusCode": 500, "headers": {"Content-Type": "application/json"}, "body": json.dumps({"error": str(e)})}

    return {"statusCode": 404, "body": "Not found"}
