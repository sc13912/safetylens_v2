import boto3, time

REGION = "ap-southeast-2"
client = boto3.client("bedrock-agentcore-control", region_name=REGION)

print("Creating AgentCore Memory with summary strategy...")

response = client.create_memory(
    name="SafetyLens_Memory",
    description="SafetyLens incident memory with auto-summarization",
    eventExpiryDuration=30,
    memoryStrategies=[
        {
            "summaryMemoryStrategy": {
                "name": "incident_summaries",
                "namespaces": ["/incidents/summaries/{sessionId}"]
            }
        }
    ]
)

memory_id = response["memory"]["id"]
print(f"Memory ID: {memory_id}")

print("Waiting for ACTIVE...")
while True:
    time.sleep(5)
    status = client.get_memory(memoryId=memory_id)["memory"]["status"]
    print(f"  Status: {status}")
    if status == "ACTIVE":
        break
    if "FAIL" in status:
        print("ERROR: Memory creation failed!")
        exit(1)

print(f"\nMemory ready: {memory_id}")
print(f"Region: {REGION}")
print(f"Strategy: summary (/incidents/summaries/{{sessionId}})")
print(f"\nCurrent memory ID: <your-memory-id>")
