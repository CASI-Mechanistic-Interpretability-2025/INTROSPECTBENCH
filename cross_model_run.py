import argparse
import sys
import os
from debug_run import run_benchmark

def main():
    parser = argparse.ArgumentParser(description="Cross-Model Evaluation Runner")
    parser.add_argument("--target", type=str, required=True, help="Target model to be introspected (e.g. nvidia/nemotron-nano-9b-v2)")
    parser.add_argument("--observers", type=str, nargs='+', required=True, help="List of observer models (e.g. openai/gpt-4o meta-llama/llama-3-8b)")
    parser.add_argument("--num_examples", type=int, default=10, help="Number of examples to run per task")
    parser.add_argument("--threads", type=int, default=1, help="Number of threads for parallel execution")
    parser.add_argument("--types", type=str, nargs='+', default=["all"], help="Specific task types to run")
    parser.add_argument("--skip_self", action="store_true", help="Skip running self-introspection (use existing cache if available)")
    parser.add_argument("--results_dir", type=str, default="results_debug", help="Directory to store results")
    
    args = parser.parse_args()
    
    print(f"Starting Cross-Model Evaluation for Target: {args.target}")
    
    # 1. Self-Introspection (Baseline & Cache Generation)
    safe_target_name = args.target.replace("/", "_")
    cache_dir = os.path.join(args.results_dir, safe_target_name, "self_introspection")
    
    if args.skip_self:
        print(f"Skipping Self-Introspection. Assuming cache exists at: {cache_dir}")
    else:
        print(f"\n{'#'*40}")
        print(f"Running Self-Introspection (Target vs Target) to generate cache...")
        print(f"{'#'*40}\n")
        
        class SelfArgs:
            model = args.target
            introspection_model = args.target # Explicitly self
            num_examples = args.num_examples
            num_threads = args.threads
            types = args.types
            # No cache_from for the self-run (it generates the data)
            cache_from = None
            results_dir = args.results_dir
            
        try:
            run_benchmark(SelfArgs)
            print("Self-Introspection completed.")
        except Exception as e:
            print(f"Failed Self-Introspection: {e}")
            # If self fails, subsequent runs might fail if they rely on cache, but we'll try to proceed or exit?
            # We'll try to proceed, maybe live generation will kick in if cache missing (as per debug_run fallback).
    
    if not os.path.exists(cache_dir):
        print(f"Warning: Cache directory {cache_dir} does not exist. Observers will use live generation.")

    # 2. Run Observers
    print(f"\nObservers to run: {args.observers}")
    
    for observer in args.observers:
        print(f"\n{'#'*40}")
        print(f"Running Observer: {observer}")
        print(f"Using Self-Introspection Cache from: {cache_dir}")
        print(f"{'#'*40}\n")
        
        class ObsArgs:
            model = args.target
            introspection_model = observer
            num_examples = args.num_examples
            num_threads = args.threads
            types = args.types
            # Use the cache we just generated (or existing)
            cache_from = cache_dir
            results_dir = args.results_dir
            
        try:
            run_benchmark(ObsArgs)
            print(f"Successfully finished Observer: {observer}")
        except Exception as e:
            print(f"Failed Observer {observer}: {e}")

if __name__ == "__main__":
    main()
