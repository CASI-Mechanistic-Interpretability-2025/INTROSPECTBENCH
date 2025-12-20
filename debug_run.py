import os
import argparse
from dotenv import load_dotenv
from benchmark.utils import OpenRouterClient
from benchmark.tasks.type1_self_pred import Task1_1_KthWord, Task1_2_PredVsCoT, Task1_4_Paraphrase
from benchmark.tasks.type2_causal import Task2_1_Subset, Task2_2_HeadsUp, Task2_3_PromptReconstruction
from benchmark.tasks.type3_state import Task3_1_ProbTargeting

# Load environment variables
load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Debug Run Introspection Benchmark")
    parser.add_argument("--model", type=str, default="nvidia/nemotron-nano-9b-v2", help="Model to test")
    parser.add_argument("--num_examples", type=int, default=10, help="Number of examples to run per task")
    parser.add_argument("--threads", type=int, default=1, help="Number of threads for parallel execution")
    parser.add_argument("--types", type=str, nargs='+', default=["all"], 
        help="Specific task types to run (default: all)")
    args = parser.parse_args()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in .env")

    print(f"Initializing OpenRouter client with model: {args.model}")
    client = OpenRouterClient(
        model_name=args.model
    )
    
    # Define tasks
    # Use model-specific output dir
    safe_model_name = args.model.replace("/", "_")
    output_dir = os.path.join("results_debug", safe_model_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Note: Dataset names must match what's expected by load_data in task_base.py
    # and correspond to files in data/ directory.
    
    # type2_prompt_reconstruction uses 'type1_open' because it needs open-ended prompts
    all_tasks = [
        Task1_1_KthWord("type1_kth_word", "debug_repo", "type1_nq", client, output_dir),
        Task1_2_PredVsCoT("type1_pred_vs_cot", "debug_repo", "type1_open", client, output_dir),
        Task1_4_Paraphrase("type1_paraphrase", "debug_repo", "type1_paraphrase", client, output_dir),
        Task2_1_Subset("type2_subset", "debug_repo", "type2_subset", client, output_dir),
        Task2_2_HeadsUp("type2_headsup", "debug_repo", "type2_headsup", client, output_dir),
        Task2_3_PromptReconstruction("type2_prompt_reconstruction", "debug_repo", "type1_open", client, output_dir),
        Task3_1_ProbTargeting("type3_prob_targeting", "debug_repo", "type3_probs", client, output_dir)
    ]

    selected_types = args.types
    if "all" in selected_types:
        tasks = all_tasks
    else:
        tasks = []
        for task in all_tasks:
            # Check if task name starts with any of the selected types
            if any(task.task_name.startswith(t) for t in selected_types):
                tasks.append(task)
    
    if not tasks:
        print(f"No tasks found matching types: {selected_types}")
        return

    for task in tasks:
        print(f"\n--- Preparing {task.task_name} ---")
        
        # Manually slice the dataset to the requested number of examples
        original_len = len(task.dataset)
        num_to_run = min(original_len, args.num_examples)
        task.dataset = task.dataset.select(range(num_to_run))
        print(f"Sliced dataset from {original_len} to {len(task.dataset)} items.")
        
        try:
            task.run(num_threads=args.threads)
            print(f"Successfully ran {task.task_name}")
        except Exception as e:
            print(f"FAILED {task.task_name}: {e}")
            import traceback
            traceback.print_exc()

    # --- Scoring ---
    print(f"\n{'='*20} Results Summary for {args.model} {'='*20}")
    import json
    import glob
    
    task_scores = {}
    
    for json_file in glob.glob(os.path.join(output_dir, "*.json")):
        task_name = os.path.basename(json_file).replace(".json", "")
        with open(json_file) as f:
            data = json.load(f)
            
        if not data:
            continue
            
        if "type1_kth_word" in task_name:
            # Accuracy of k=1
            correct = sum(1 for item in data if item["metrics"].get("k=1", {}).get("correct", False))
            score = correct / len(data) if data else 0
            task_scores[task_name] = score
            print(f"{task_name}: {score:.2%} (Exact Match k=1)")
            
        elif "type1_pred_vs_cot" in task_name:
            # Shift error (lower is better). Score = 1 - avg_shift
            avg_shift = sum(item["diff_shift"] for item in data) / len(data) if data else 0
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
            # Success rate
            success = sum(1 for item in data if item.get("success", False))
            score = success / len(data) if data else 0
            task_scores[task_name] = score
            print(f"{task_name}: {score:.2%} (Success Rate)")
            
        elif "type2_prompt_reconstruction" in task_name:
            # Semantic Similarity (Avg)
            avg_sim = sum(item.get("similarity_score", 0.0) for item in data) / len(data) if data else 0
            task_scores[task_name] = avg_sim
            print(f"{task_name}: {avg_sim:.4f} (Avg Cos Similarity)")
            
        elif "type3_prob_targeting" in task_name:
            pass
            
    if task_scores:
        final_score = sum(task_scores.values()) / len(task_scores)
        print(f"\nFinal Aggregated Score: {final_score:.2%} (Average of normalized task scores)")
    else:
        print("No scores available.")


if __name__ == "__main__":
    main()
