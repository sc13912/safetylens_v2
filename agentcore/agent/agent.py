import os, logging, boto3
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from strands import Agent
from strands.telemetry import StrandsTelemetry
from strands.models import BedrockModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REGION = os.getenv("AWS_DEFAULT_REGION", "ap-southeast-2")
MEMORY_ID = os.getenv("MEMORY_ID")
MODEL_ID = os.getenv("MODEL_ID", "au.anthropic.claude-haiku-4-5-20251001-v1:0")

# Initialize OTEL exporter for Langfuse
strands_telemetry = StrandsTelemetry()
strands_telemetry.setup_otlp_exporter()

app = BedrockAgentCoreApp()
memory_client = boto3.client("bedrock-agentcore", region_name=REGION)

SYSTEM_PROMPT = """You are a warehouse safety insights agent for SafetyLens.
You have access to a memory store of safety incidents detected by an AI-powered CCTV system.
Each incident contains: a safety score (0-100), hazard deductions (type, points, detail), and a summary.

When answering questions:
- Search memory for relevant incidents
- Provide specific data (scores, hazard types, timestamps)
- Be concise and actionable
- If asked about trends, summarize patterns across multiple incidents
"""

model = BedrockModel(model_id=MODEL_ID, region_name=REGION, temperature=0.1, max_tokens=1024)
agent = Agent(model=model, system_prompt=SYSTEM_PROMPT)


@app.entrypoint
def invoke(payload, context=None):
    prompt = payload.get("prompt", "Hello")
    logger.info(f"Query: {prompt}")

    context_str = ""
    if MEMORY_ID:
        try:
            response = memory_client.retrieve_memory_records(
                memoryId=MEMORY_ID,
                namespace="/incidents/summaries/cctv-monitoring",
                searchCriteria={"searchQuery": prompt, "topK": 15}
            )
            records = response.get("memoryRecordSummaries", [])
            if records:
                context_str = "\n\nRelevant safety incidents from memory:\n"
                for r in records:
                    context_str += f"- {r.get('content', {}).get('text', '')}\n"
        except Exception as e:
            logger.warning(f"Memory search failed: {e}")

    full_prompt = prompt + context_str if context_str else prompt
    response = agent(full_prompt)
    return response.message["content"][0]["text"]


if __name__ == "__main__":
    app.run()
