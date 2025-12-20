import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

try:
    print("Sending request...")
    resp = client.chat.completions.create(
        model="qwen/qwen-2.5-72b-instruct",
        messages=[{"role": "user", "content": "Explain why 2+2=4 step by step."}],
        max_tokens=200
    )
    print("--- RAW RESPONSE ---")
    print(resp)
    print("--- CONTENT ---")
    print(resp.choices[0].message.content)
except Exception as e:
    print(f"Error: {e}")
