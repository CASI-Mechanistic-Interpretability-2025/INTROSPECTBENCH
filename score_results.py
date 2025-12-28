import os
import json
import glob
import argparse
from prettytable import PrettyTable

def calculate_task_score(task_name, data):
    if not data: return 0.0, "N/A"
    
    score = 0.0
    metric_str = ""

    if "type1_kth_word" in task_name:
        correct = sum(1 for item in data if item["metrics"].get("k=1", {}).get("correct", False))
        score = correct / len(data)
        metric_str = "Exact Match k=1"
        
    elif "type1_pred_vs_cot" in task_name:
        avg_shift = sum(item["diff_shift"] for item in data) / len(data)
        score = max(0, 1.0 - avg_shift)
        metric_str = f"1 - AvgShift[{avg_shift:.2f}]"
        
    elif "type1_paraphrase" in task_name:
        correct = sum(1 for item in data if item.get("is_correct", False))
        score = correct / len(data)
        metric_str = "Accuracy"

    elif "type2_subset" in task_name or "type2_headsup" in task_name:
        success = sum(1 for item in data if item.get("success", False))
        score = success / len(data)
        metric_str = "Success Rate"
        
    elif "type2_prompt_reconstruction" in task_name:
        avg_sim = sum(item.get("similarity_score", 0.0) for item in data) / len(data)
        score = avg_sim
        metric_str = "Avg Cos Similarity"
    
    return score, metric_str

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Target model to analyze (Legacy alias for --target)")
    parser.add_argument("--target", type=str, help="Target model (introspected)")
    parser.add_argument("--observer", type=str, help="Observer model (introspector). If not set, processes all observers for the target.")
    parser.add_argument("--results_dir", type=str, default="results_debug", help="Base directory containing results (default: results_debug)")
    parser.add_argument("--output_csv", type=str, help="Path to save CSV summary")
    args = parser.parse_args()
    
    # accept legacy argument
    target_model = args.target if args.target else args.model
    if not target_model:
        print("Error: Must specify --target (or --model)")
        return
        
    safe_target_name = target_model.replace("/", "_")
    base_dir = os.path.join(args.results_dir, safe_target_name)
    
    if not os.path.exists(base_dir):
        print(f"No results found for target: {base_dir}")
        return

    observers = []
    if args.observer:
        observers = [args.observer.replace("/", "_")]
    else:
        # Scan subdirectories
        try:
             observers = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        except:
             pass
    
    if not observers:
        print(f"No observer directories found in {base_dir}")
        return

    print(f"\n{'='*20} Results for Target: {target_model} {'='*20}")
    
    # We will build a table: Observer | Task1 | Task2 ... | Average
    # First find all unique tasks across all observers
    all_tasks = set()
    observer_results = {} # {observer: {task: score}}

    for obs in observers:
        obs_dir = os.path.join(base_dir, obs)
        observer_results[obs] = {}
        json_files = glob.glob(os.path.join(obs_dir, "*.json"))
        
        for json_file in json_files:
            task_name = os.path.basename(json_file).replace(".json", "")
            all_tasks.add(task_name)
            
            with open(json_file) as f:
                try:
                    data = json.load(f)
                    score, _ = calculate_task_score(task_name, data)
                    observer_results[obs][task_name] = score
                except:
                    print(f"Error loading {json_file}")
                    observer_results[obs][task_name] = 0.0
    
    sorted_tasks = sorted(list(all_tasks))
    
    table = PrettyTable()
    table.field_names = ["Observer"] + sorted_tasks + ["Average"]
    
    for obs in sorted(observers):
        row = [obs]
        scores = []
        for task in sorted_tasks:
            s = observer_results[obs].get(task, "N/A")
            if s != "N/A":
                scores.append(s)
                row.append(f"{s:.2%}")
            else:
                row.append("-")
        
        if scores:
            avg_score = sum(scores) / len(scores)
            row.append(f"{avg_score:.2%}")
        else:
            row.append("0.00%")
            
        table.add_row(row)
        
    print(table)
    
    if args.output_csv:
        with open(args.output_csv, "w") as f:
            f.write(table.get_csv_string())
            print(f"Saved CSV to {args.output_csv}")

if __name__ == "__main__":
    main()
