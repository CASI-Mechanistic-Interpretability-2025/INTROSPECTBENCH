import argparse
import os
from benchmark.utils import OpenRouterClient
# Import Tasks
from benchmark.tasks.type1_self_pred import Task1_1_KthWord, Task1_2_PredVsCoT, Task1_3_SelfRecognition
from benchmark.tasks.type2_causal import Task2_1_Subset, Task2_2_HeadsUp, Task2_3_Deception
from benchmark.tasks.type3_state import Task3_1_ProbTargeting
from dotenv import load_dotenv

load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Run Introspection Benchmark")
    parser.add_argument("--task", type=str, required=True, 
                        help="Task to run: type1_kth, type1_cot, type1_recog, type2_subset, type2_headsup, type2_deception, type3_prob, or 'all'")
    parser.add_argument("--model", type=str, default="openai/gpt-4o-mini", help="Model name for OpenRouter (e.g. openai/gpt-4o-mini)")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save results")
    
    args = parser.parse_args()
    
    client = OpenRouterClient(model_name=args.model)
    output_dir = os.path.join(args.output_dir, args.model.replace("/", "_"))
    
    # repo_id is no longer needed as we load locally
    repo_id_placeholder = "local"
    
    tasks = []
    
    # Task Registry
    if args.task in ["type1_kth", "all"]:
        tasks.append(Task1_1_KthWord("type1_kth_word", repo_id_placeholder, "type1_nq", client, output_dir))
        
    if args.task in ["type1_cot", "all"]:
        tasks.append(Task1_2_PredVsCoT("type1_cot_forced", repo_id_placeholder, "type1_open", client, output_dir))
        
    if args.task in ["type1_recog", "all"]:
        tasks.append(Task1_3_SelfRecognition("type1_self_recognition", repo_id_placeholder, "type1_open", client, output_dir))
        
    if args.task in ["type2_subset", "all"]:
        tasks.append(Task2_1_Subset("type2_subset", repo_id_placeholder, "type2_subset", client, output_dir))
        
    if args.task in ["type2_headsup", "all"]:
        tasks.append(Task2_2_HeadsUp("type2_headsup", repo_id_placeholder, "type2_headsup", client, output_dir))
        
    if args.task in ["type2_deception", "all"]:
        tasks.append(Task2_3_Deception("type2_deception", repo_id_placeholder, "type2_deception", client, output_dir))
        
    if args.task in ["type3_prob", "all"]:
        tasks.append(Task3_1_ProbTargeting("type3_prob_targeting", repo_id_placeholder, "type3_probs", client, output_dir))

    if not tasks:
        print(f"No valid tasks found for argument: {args.task}")
        return

    print(f"Starting benchmark run for model {args.model}...")
    for task in tasks:
        try:
            task.run()
        except Exception as e:
            print(f"Task {task.task_name} failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
