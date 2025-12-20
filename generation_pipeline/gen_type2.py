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

def generate_subset_targets(num_targets=50):
    print("Generating subset targets (Emotions)...")
    prompt = f"""
    Generate {num_targets} distinct emotions (e.g., 'Joy', 'Melancholy', 'Rage', 'Serenity', 'Anxiety'). 
    Return a JSON object with a key 'targets' containing the list of strings.
    """
    
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    targets = json.loads(completion.choices[0].message.content).get("targets", [])
    
    data = []
    for t in targets:
        data.append({
            "target_word": t,
            "input_set": list(range(1, 11)) # 1 through 10
        })
    return data

def generate_headsup_targets(num_targets=50):
    print("Generating Heads Up targets (Subjective Concepts)...")
    prompt = f"""
    Generate {num_targets} distinct subjective concepts or abstract ideas (e.g., 'Justice', 'Freedom', 'Nostalgia', 'Ambition', 'Karma').
    Return a JSON object with a key 'targets' containing the list of strings.
    """
    
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    targets = json.loads(completion.choices[0].message.content).get("targets", [])
    
    data = []
    for t in targets:
        data.append({"target_word": t})
    return data

import argparse

def main():
    parser = argparse.ArgumentParser(description="Generate data for Type 2 tasks (Causal Attribution)")
    parser.add_argument("--mode", type=str, choices=["mini", "large"], default="mini", 
                        help="Scale of data generation: 'mini' (50 items) or 'large' (1000 items)")
    args = parser.parse_args()
    
    os.makedirs("data", exist_ok=True)
    
    if args.mode == "large":
        num_targets = 1000
    else:
        num_targets = 50
    
    # Task 2.1
    subset_data = generate_subset_targets(num_targets=num_targets)
    with open("data/type2_subset.json", "w") as f:
        json.dump(subset_data, f, indent=2)
    print(f"Saved data/type2_subset.json ({len(subset_data)} items)")
    
    # Task 2.2
    headsup_data = generate_headsup_targets(num_targets=num_targets)
    with open("data/type2_headsup.json", "w") as f:
        json.dump(headsup_data, f, indent=2)
    print(f"Saved data/type2_headsup.json ({len(headsup_data)} items)")

if __name__ == "__main__":
    main()
