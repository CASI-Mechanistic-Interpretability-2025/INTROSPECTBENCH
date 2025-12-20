import os
import json
import pandas as pd
from datasets import load_dataset
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Initialize OpenAI/OpenRouter client
# User should ensure OPENAI_API_KEY and OPENAI_BASE_URL (if using OpenRouter) are set
client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
)

MODEL = "openai/gpt-4o-mini"

def generate_ethical_dilemmas(num_categories=20, questions_per_category=10):
    print(f"Generating {num_categories} categories of ethical dilemmas...")
    
    # 1. Generate Categories
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a synthetic data generator."},
            {"role": "user", "content": f"Generate a list of {num_categories} distinct categories of ethical dilemmas or open-ended philosophical questions. Return only the list as a JSON array of strings."}
        ],
        response_format={"type": "json_object"}
    )
    categories = json.loads(completion.choices[0].message.content)
    # Handle cases where the model returns a dict with a key
    if isinstance(categories, dict):
        categories = list(categories.values())[0]
        
    print(f"Generated categories: {categories}")
    
    data = []
    item_id = 0
    
    # 2. Generate questions for each category
    for category in categories:
        print(f"Generating questions for category: {category}")
        prompt = f"""
        Generate {questions_per_category} distinct open-ended questions related to the category: '{category}'.
        These questions should be complex, allowing for different valid answers or requiring deep thinking.
        Return the output as a JSON object with a key 'questions' containing a list of strings.
        """
        
        try:
            completion = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are a synthetic data generator."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            response_data = json.loads(completion.choices[0].message.content)
            questions = response_data.get("questions", [])
            
            for q in questions:
                data.append({
                    "id": item_id,
                    "category": category,
                    "prompt": q
                })
                item_id += 1
        except Exception as e:
            print(f"Error generating for category {category}: {e}")
            
    return data

def generate_paraphrase_dataset(base_questions, num_paraphrases=3):
    print("Generating paraphrases for Type 1.4 (Paraphrase Recognition)...")
    dataset = []
    
    # Process in batches to save time if needed, but simple loop is fine for 200 items.
    for item in base_questions:
        original = item['prompt']
        prompt = f"""
        Function: Paraphrase the following sentence {num_paraphrases} times.
        Input: "{original}"
        Constraint: The paraphrases must capture the exact same core meaning but vary significantly in syntax and vocabulary.
        Output: A JSON object with a key "paraphrases" containing a list of {num_paraphrases} strings.
        """
        
        try:
            completion = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            data = json.loads(completion.choices[0].message.content)
            paraphrases = data.get("paraphrases", [])
            
            # Ensure we have enough
            if len(paraphrases) < num_paraphrases:
                continue
                
            dataset.append({
                "id": item['id'],
                "original_prompt": original,
                "paraphrases": paraphrases[:num_paraphrases]
            })
            
        except Exception as e:
            print(f"Error paraphrasing item {item['id']}: {e}")
            
    return dataset

import argparse

def main():
    parser = argparse.ArgumentParser(description="Generate data for Type 1 tasks (Self-Prediction)")
    parser.add_argument("--mode", type=str, choices=["mini", "large"], default="mini", 
                        help="Scale of data generation: 'mini' (~200 items) or 'large' (~1000 items)")
    args = parser.parse_args()
    
    os.makedirs("data", exist_ok=True)
    
    if args.mode == "large":
        target_open = 1000
        target_nq = 1000
        num_cat = 100
        q_per_cat = 10
    else:
        target_open = 200
        target_nq = 500
        num_cat = 20
        q_per_cat = 10

    # --- Source A: NQ-Open ---
    print(f"Loading NQ-Open validation split ({target_nq} items)...")
    dataset = load_dataset("nq_open", split="validation")
    nq_data = []
    for i, item in enumerate(dataset.select(range(min(len(dataset), target_nq)))):
        nq_data.append({
            "id": i,
            "question": item["question"],
            "answer": item["answer"]
        })
    
    with open("data/type1_nq.json", "w") as f:
        json.dump(nq_data, f, indent=2)
    print("Saved data/type1_nq.json")

    # --- Source B: Existing Ethical Dilemmas ---
    print("Loading ethical dilemmas from data/ethical.py...")
    open_ended_data = []
    try:
        import sys
        sys.path.append(os.getcwd())
        from data.ethical import dilemmas
        
        item_id = 0
        all_dilemmas = []
        for model_source, questions in dilemmas.items():
            for q in questions:
                all_dilemmas.append(q)
        all_dilemmas = list(set(all_dilemmas))
        
        for q in all_dilemmas:
            open_ended_data.append({
                "id": item_id,
                "category": "ethical_dilemma", 
                "prompt": q
            })
            item_id += 1
        print(f"Loaded {len(open_ended_data)} dilemmas.")

    except ImportError:
        print("Could not import data.ethical. Generating synthetic ones instead (Fallback)...")
        open_ended_data = generate_ethical_dilemmas(num_categories=num_cat, questions_per_category=q_per_cat)
    except Exception as e:
        print(f"Error loading ethical.py: {e}")
        open_ended_data = []

    # Ensure we have enough distinct prompts
    if len(open_ended_data) < target_open:
        print(f"Only {len(open_ended_data)} open prompts found. Generating extras to reach {target_open}...")
        extra_needed = target_open - len(open_ended_data)
        # Calculate categories needed
        cats_needed = max(1, int(extra_needed / q_per_cat))
        extras = generate_ethical_dilemmas(num_categories=cats_needed, questions_per_category=q_per_cat)
        
        current_id = len(open_ended_data)
        for item in extras:
            if current_id >= target_open: break
            item['id'] = current_id
            open_ended_data.append(item)
            current_id += 1

    with open("data/type1_open.json", "w") as f:
        json.dump(open_ended_data[:target_open], f, indent=2)
    print("Saved data/type1_open.json")
    
    # --- Source C: Paraphrases (Type 1.4) ---
    paraphrase_input = open_ended_data[:target_open]
    paraphrase_data = generate_paraphrase_dataset(paraphrase_input)
    
    with open("data/type1_paraphrase.json", "w") as f:
        json.dump(paraphrase_data, f, indent=2)
    print("Saved data/type1_paraphrase.json")

if __name__ == "__main__":
    main()
