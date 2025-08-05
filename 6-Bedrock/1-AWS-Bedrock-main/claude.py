# import boto3
# import json

# prompt_data="""
# Act as a Shakespeare and write a poem on Genertaive AI
# """

# bedrock=boto3.client(service_name="bedrock-runtime")

# payload={
#     "prompt":prompt_data,
#     "max_tokens_to_sample":512,
#     "temperature":0.8,
#     # "topP":0.8
# }
# body = json.dumps(payload)
# model_id = "us.anthropic.claude-3-5-haiku-20241022-v1:0"
# response = bedrock.invoke_model(
#     body=body,
#     modelId=model_id,
#     accept="application/json",
#     contentType="application/json",
# )

# response_body = json.loads(response.get("body").read())
# response_text = response_body.get("completions")[0].get("data").get("text")
# print(response_text)
import boto3
import json

bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

raw = "Act as Shakespeare and write a poem on Generative AI."

payload = {
    "anthropic_version": "bedrock-2023-05-31",     # required
    "system": "You are William Shakespeare.",      # top‑level system prompt
    "messages": [                                  # only user/assistant roles here
        {"role": "user", "content": raw}
    ],
    "max_tokens": 512,                             # from AWS example :contentReference[oaicite:0]{index=0}
    "temperature": 0.8,
    "top_p": 0.9
}

response = bedrock.invoke_model(
    modelId="us.anthropic.claude-opus-4-20250514-v1:0",
    body=json.dumps(payload),
    contentType="application/json",
    accept="application/json"
)

resp = json.loads(response["body"].read())
# iterate through each content block and print all the text pieces
for block in resp.get("content", []):
    if block.get("type") == "text":
        print(block.get("text"), end="")

# (optionally newline at end)
print()
