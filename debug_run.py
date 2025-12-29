import os
import argparse
from dotenv import load_dotenv
from benchmark.utils import OpenRouterClient, CacheClient
from benchmark.tasks.type1_self_pred import Task1_1_KthWord, Task1_2_PredVsCoT, Task1_3_SelfRecognition, Task1_4_Paraphrase
from benchmark.tasks.type2_causal import Task2_1_Subset, Task2_2_HeadsUp, Task2_3_PromptReconstruction
from benchmark.tasks.type3_state import Task3_1_ProbTargeting

# Map string arg to Class
TASK_REGISTRY = {
    "type1_kth_word": Task1_1_KthWord,
    "type1_pred_vs_cot": Task1_2_PredVsCoT,
    "type1_self_recognition": Task1_3_SelfRecognition,
    "type1_stype1_paraphrase": Task1_4_Paraphrase, # Typo in user code? Assuming meant Task1_4
    "type1_paraphrase": Task1_4_Paraphrase,
    "type2_subset": Task2_1_Subset,
    "type2_headsup": Task2_2_HeadsUp,
    "type2_prompt_reconstruction": Task2_3_PromptReconstruction,
    "type3_prob_targeting": Task3_1_ProbTargeting,
    "type3_probs": Task3_1_ProbTargeting # Alias
}

def run_benchmark(args):
    # Load environment variables
    load_dotenv()
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in .env file")
        
    print(f"Initializing OpenRouter client with target model: {args.model}")
    client = OpenRouterClient(model_name=args.model)
    
    client_introspection = None
    if args.introspection_model:
        if args.introspection_model == args.model:
             print("Using target model for introspection (Self-Introspection Explicit).")
             client_introspection = client
        else:
             print(f"Initializing OpenRouter client with introspection model: {args.introspection_model}")
             client_introspection = OpenRouterClient(model_name=args.introspection_model)
    else:
        # Default to self-introspection if not specified
        print("Using target model for introspection (Self-Introspection Default).")
        client_introspection = client
    
    # Define tasks
    # Use model-specific output dir based on Cross-Model Plan
    # {results_dir}/{target}/{observer}
    
    results_base = getattr(args, "results_dir", "results_debug")
    safe_target_name = args.model.replace("/", "_")
    
    if args.introspection_model and args.introspection_model != args.model:
        safe_observer_name = args.introspection_model.replace("/", "_")
        output_dir = os.path.join(results_base, safe_target_name, safe_observer_name)
    else:
        # Self-Introspection
        output_dir = os.path.join(results_base, safe_target_name, "self_introspection")
        
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving results to: {output_dir}")
    
    tasks_to_run = []
    
    # Handle "all" or specific types
    selected_types = args.types if args.types else ["all"]
    if isinstance(selected_types, str): selected_types = [selected_types] # Ensure list
    if "all" in selected_types:
        standard_tasks = ["type1_kth_word", "type1_pred_vs_cot", "type1_paraphrase", 
                         "type2_subset", "type2_headsup", "type2_prompt_reconstruction",
                         "type3_prob_targeting"]
        for t in standard_tasks:
            tasks_to_run.append((t, TASK_REGISTRY[t]))
    else:
        for t_name in selected_types:
            if t_name in TASK_REGISTRY:
                tasks_to_run.append((t_name, TASK_REGISTRY[t_name]))
            elif any(key.startswith(t_name) for key in TASK_REGISTRY.keys()):
                 for key in TASK_REGISTRY.keys():
                     if key.startswith(t_name):
                         tasks_to_run.append((key, TASK_REGISTRY[key]))
            else:
                print(f"Warning: Task {t_name} not found in registry.")

    # Remove duplicates
    tasks_to_run = list(set(tasks_to_run))

    for task_name, TaskClass in tasks_to_run:
        print(f"\n--- Preparing {task_name} ---")
        
        # Determine Target Client (Cached or Live)
        target_client_for_task = client
        if hasattr(args, 'cache_from') and args.cache_from:
            cache_file = os.path.join(args.cache_from, f"{task_name}.json")
            if os.path.exists(cache_file):
                print(f"Using CACHED target responses from: {cache_file}")
                try:
                    target_client_for_task = CacheClient(cache_file)
                except Exception as e:
                    print(f"Error loading cache: {e}. Falling back to live generation.")
            else:
                print(f"Cache file not found at {cache_file}. Using live generation.")

        # Load data
        if task_name == "type1_kth_word": data_file = "type1_nq.json"
        elif task_name == "type1_pred_vs_cot": data_file = "type1_open.json"
        elif task_name == "type1_self_recognition": data_file = "type1_open.json" # Assuming same dataset as open/pred_vs_cot? Let's check logic or default
        elif task_name == "type1_paraphrase": data_file = "type1_paraphrase.json"
        elif task_name == "type2_subset": data_file = "type2_subset.json"
        elif task_name == "type2_headsup": data_file = "type2_headsup.json"
        elif task_name == "type2_prompt_reconstruction": data_file = "type1_open.json"
        elif task_name == "type3_prob_targeting": data_file = "type3_probs.json"
        else: data_file = f"{task_name}.json"
        
        data_path = os.path.join("data", data_file)
        
        # Fallback if file not found (try exact name)
        if not os.path.exists(data_path):
             alt_path = os.path.join("data", f"{task_name}.json")
             if os.path.exists(alt_path):
                 data_path = alt_path
        
        # Task init signature: (task_name, dataset_name, dataset_split, client_target, client_introspection, output_dir)
        split_name = data_file.replace(".json", "")
        
        task_instance = TaskClass(
            task_name, 
            "debug_repo", 
            split_name, 
            target_client_for_task, 
            client_introspection, 
            output_dir
        )

        try:
            if args.num_examples:
                if hasattr(task_instance, 'dataset'):
                    original_len = len(task_instance.dataset)
                    task_instance.dataset = task_instance.dataset.select(range(min(original_len, args.num_examples)))
                    print(f"Sliced dataset from {original_len} to {len(task_instance.dataset)} items.")
            
            task_instance.run(num_threads=args.num_threads)
            print(f"Successfully ran {task_name}")
        except Exception as e:
            print(f"Error running {task_name}: {e}")
            import traceback
            traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="Debug Run Introspection Benchmark")
    parser.add_argument("--model", type=str, required=True, help="Target model (e.g. nvidia/nemotron-nano-9b-v2)")
    parser.add_argument("--introspection_model", type=str, default=None, help="Model to use for introspection (Guesser). Defaults to target model if not set.")
    parser.add_argument("--num_examples", type=int, default=10, help="Number of examples to run per task")
    parser.add_argument("--threads", type=int, default=1, help="Number of threads for parallel execution")
    parser.add_argument("--num_threads", type=int, dest="num_threads", help="Alias for --threads") 
    parser.add_argument("--cache_from", type=str, default=None, help="Directory to load cached target responses from (e.g. results_debug/model/self_introspection)")
    parser.add_argument("--results_dir", type=str, default="results_debug", help="Directory to save results")
    
    parser.add_argument("--types", type=str, nargs='+', default=["all"], 
        help="Specific task types to run (default: all)")
        
    args = parser.parse_args()
    
    if args.threads and not args.num_threads:
        args.num_threads = args.threads
        
    run_benchmark(args)

if __name__ == "__main__":
    main()
