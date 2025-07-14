# import boto3
# import json
# import base64
# import os

# prompt_data = """
# provide me an 4k hd image of a beach, also use a blue sky rainy season and
# cinematic display
# """
# prompt_template=[{"text":prompt_data,"weight":1}]
# bedrock = boto3.client(service_name="bedrock-runtime")
# payload = {
#     "text_prompts":prompt_template,
#     "cfg_scale": 10,
#     "seed": 0,
#     "steps":50,
#     "width":512,
#     "height":512

# }

# body = json.dumps(payload)
# model_id = "stability.stable-diffusion-xl-v0"
# response = bedrock.invoke_model(
#     body=body,
#     modelId=model_id,
#     accept="application/json",
#     contentType="application/json",
# )

# response_body = json.loads(response.get("body").read())
# print(response_body)
# artifact = response_body.get("artifacts")[0]
# image_encoded = artifact.get("base64").encode("utf-8")
# image_bytes = base64.b64decode(image_encoded)

# # Save image to a file in the output directory.
# output_dir = "output"
# os.makedirs(output_dir, exist_ok=True)
# file_name = f"{output_dir}/generated-img.png"
# with open(file_name, "wb") as f:
#     f.write(image_bytes)



import base64
import boto3
import json
import os
import random

# 1. Create a Bedrock Runtime client
bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

# 2. Choose your prompt and a random seed
prompt = "A 4K HD cinematic beach scene, rainy season under a blue sky"
seed = random.randint(0, 4294967295)

# 3. Build the payload according to the native schema :contentReference[oaicite:0]{index=0}
payload = {
    "text_prompts": [{"text": prompt}],   # your prompt(s)
    "seed": seed,                         # 32‑bit integer
    "cfg_scale": 10,                      # guidance scale
    "steps": 50,                          # diffusion steps
    # optional style presets: "photographic", "cinematic", etc.
    # "style_preset": "cinematic",
    # width/height are *not* required for XL v1—omit unless your account docs show support
}

# 4. Invoke the model
response = bedrock.invoke_model(
    modelId="stability.stable-diffusion-xl-v1",  # use XL v1 for “stability.stable-diffusion-xl-v0”
    body=json.dumps(payload),
    contentType="application/json",
    accept="application/json"
)

# 5. Parse out the base64 artifact :contentReference[oaicite:1]{index=1}
model_response = json.loads(response["body"].read())
artifact = model_response["artifacts"][0]
img_data = base64.b64decode(artifact["base64"])

# 6. Save to disk
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
file_path = os.path.join(output_dir, "generated-beach.png")
with open(file_path, "wb") as f:
    f.write(img_data)

print(f"Saved image to {file_path}")
