import os
import json
import random
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
)

MODEL = "openai/gpt-4o-mini"

def generate_prob_categories(num_categories=100):
    print("Generating Probability Target categories...")
    prompt = f"""
    Generate {num_categories} distinct topics or categories for text generation (e.g., 'A specific color', 'A type of fruit', 'Comparison of two cities').
    Return a JSON object with a key 'categories' containing the list of strings.
    """
    
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    categories = json.loads(completion.choices[0].message.content).get("categories", [])
    
    data = []
    for cat in categories:
        data.append({
            "category": cat,
            "target_probs": [0.5, 0.9, 1.0]
        })
    return data

import argparse

def main():
    parser = argparse.ArgumentParser(description="Generate data for Type 3 tasks (State Introspection)")
    parser.add_argument("--mode", type=str, choices=["mini", "large"], default="mini", 
                        help="Scale of data generation: 'mini' (10 items) or 'large' (1000 items)")
    args = parser.parse_args()
    
    os.makedirs("data", exist_ok=True)
    
    if args.mode == "large":
        num_categories = 1000
    else:
        num_categories = 10 # Mini
    
    type3_data = generate_prob_categories(num_categories=num_categories)
    with open("data/type3_probs.json", "w") as f:
        json.dump(type3_data, f, indent=2)
    print(f"Saved data/type3_probs.json ({len(type3_data)} items)")

if __name__ == "__main__":
    main()
