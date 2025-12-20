import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    # try reading from .env manually if load_dotenv failed or env not set
    try:
        with open(".env") as f:
            for line in f:
                if line.startswith("OPENROUTER_API_KEY"):
                    api_key = line.split("=", 1)[1].strip().strip('"')
                    break
    except:
        pass

if not api_key:
    print("API Key not found")
    exit(1)

resp = requests.get("https://openrouter.ai/api/v1/models", headers={"Authorization": f"Bearer {api_key}"})
if resp.status_code == 200:
    data = resp.json()
    qwen_models = [m["id"] for m in data["data"] if "qwen" in m["id"].lower()]
    for m in sorted(qwen_models):
        print(m)
else:
    print(f"Error: {resp.status_code}")
    print(resp.text)
