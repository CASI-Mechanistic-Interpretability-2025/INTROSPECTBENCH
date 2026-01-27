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

    elif "type1_self_recognition" in task_name:
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
    parser.add_argument("--target", type=str, help="Target model (introspected). If not set, processes all targets found in results_dir.")
    parser.add_argument("--observer", type=str, help="Observer model (introspector). If not set, processes all observers for the target.")
    parser.add_argument("--results_dir", type=str, default="results_debug", help="Base directory containing results (default: results_debug)")
    parser.add_argument("--output_csv", type=str, help="Path to save CSV summary")
    args = parser.parse_args()
    
    # accept legacy argument
    specified_target = args.target if args.target else args.model
    
    targets_to_process = []
    if specified_target:
        # User specified a target, convert to safe name for directory lookup
        safe_name = specified_target.replace("/", "_")
        targets_to_process.append((specified_target, safe_name))
    elif os.path.exists(args.results_dir):
        # Scan all directories
        for d in os.listdir(args.results_dir):
            if os.path.isdir(os.path.join(args.results_dir, d)):
                targets_to_process.append((d, d)) # Display name is same as dir name for scanned
    
    if not targets_to_process:
        print(f"No targets found in {args.results_dir}")
        return

    # We will build a table: Target | Observer | Task1 | Task2 ... | Average
    all_tasks = set()
    # Structure: results[target_display][observer] = {task: score}
    results = {} 

    for target_display, target_safe_name in targets_to_process:
        base_dir = os.path.join(args.results_dir, target_safe_name)
        if not os.path.exists(base_dir):
            if specified_target: print(f"No results found for target: {base_dir}")
            continue
            
        results[target_display] = {}
        
        # Find observers
        current_observers = []
        if args.observer:
             current_observers = [args.observer.replace("/", "_")]
        else:
             try:
                 current_observers = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
             except:
                 pass
        
        for obs in current_observers:
            obs_dir = os.path.join(base_dir, obs)
            if not os.path.exists(obs_dir): continue
            
            results[target_display][obs] = {}
            json_files = glob.glob(os.path.join(obs_dir, "*.json"))
            
            for json_file in json_files:
                task_name = os.path.basename(json_file).replace(".json", "")
                all_tasks.add(task_name)
                
                with open(json_file) as f:
                    try:
                        data = json.load(f)
                        score, _ = calculate_task_score(task_name, data)
                        results[target_display][obs][task_name] = score
                    except:
                        # print(f"Error loading {json_file}")
                        results[target_display][obs][task_name] = 0.0

    sorted_tasks = sorted(list(all_tasks))
    
    table = PrettyTable()
    table.field_names = ["Target", "Observer"] + sorted_tasks + ["Average"]
    
    for target in sorted(results.keys()):
        for obs in sorted(results[target].keys()):
            row = [target, obs]
            scores = []
            for task in sorted_tasks:
                s = results[target][obs].get(task, "N/A")
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
