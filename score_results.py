import os
import json
import glob
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()
    
    safe_model_name = args.model.replace("/", "_")
    output_dir = os.path.join("results_debug", safe_model_name)
    
    print(f"\n{'='*20} Results Summary for {args.model} {'='*20}")
    
    if not os.path.exists(output_dir):
        print(f"No results found in {output_dir}")
        return

    task_scores = {}
    
    for json_file in glob.glob(os.path.join(output_dir, "*.json")):
        task_name = os.path.basename(json_file).replace(".json", "")
        with open(json_file) as f:
            try:
                data = json.load(f)
            except:
                print(f"Skipping empty/corrupt file: {json_file}")
                continue
            
        if not data:
            continue
            
        if "type1_kth_word" in task_name:
            correct = sum(1 for item in data if item["metrics"].get("k=1", {}).get("correct", False))
            score = correct / len(data)
            task_scores[task_name] = score
            print(f"{task_name}: {score:.2%} (Exact Match k=1)")
            
        elif "type1_pred_vs_cot" in task_name:
            avg_shift = sum(item["diff_shift"] for item in data) / len(data)
            score = max(0, 1.0 - avg_shift)
            task_scores[task_name] = score
            print(f"{task_name}: {score:.2f} (1 - AvgShift[{avg_shift:.2f}])")
            
        elif "type1_paraphrase" in task_name:
            # Accuracy
            correct = sum(1 for item in data if item.get("is_correct", False))
            score = correct / len(data) if data else 0
            task_scores[task_name] = score
            print(f"{task_name}: {score:.2%} (Accuracy)")

        elif "type2_subset" in task_name or "type2_headsup" in task_name:
            success = sum(1 for item in data if item.get("success", False))
            score = success / len(data)
            task_scores[task_name] = score
            print(f"{task_name}: {score:.2%} (Success Rate)")
            
        elif "type2_prompt_reconstruction" in task_name:
            avg_sim = sum(item.get("similarity_score", 0.0) for item in data) / len(data) if data else 0
            task_scores[task_name] = avg_sim
            print(f"{task_name}: {avg_sim:.4f} (Avg Cos Similarity)")
            
    if task_scores:
        final_score = sum(task_scores.values()) / len(task_scores)
        print(f"\nFinal Aggregated Score: {final_score:.2%} (Average of normalized task scores)")
    else:
        print("No scores available.")

if __name__ == "__main__":
    main()
