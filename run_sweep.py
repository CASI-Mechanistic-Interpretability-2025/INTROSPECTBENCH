import subprocess
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Sweep Introspection Benchmark over multiple models")
    parser.add_argument("--models", type=str, nargs='+', required=True, help="List of model names to test")
    parser.add_argument("--num_examples", type=int, default=10, help="Number of examples per task")
    parser.add_argument("--threads", type=int, default=5, help="Number of threads per task")
    parser.add_argument("--types", type=str, nargs='+', default=["all"], help="Task types to run")
    
    args = parser.parse_args()
    
    for model in args.models:
        print(f"\n{'='*30}")
        print(f"STARTING BENCHMARK FOR MODEL: {model}")
        print(f"{'='*30}\n")
        
        cmd = [
            "python", "debug_run.py",
            "--model", model,
            "--num_examples", str(args.num_examples),
            "--threads", str(args.threads),
        ]
        if args.types:
            cmd.extend(["--types"] + args.types)
            
        try:
            subprocess.run(cmd, check=True)
            print(f"\nCOMPLETED BENCHMARK FOR MODEL: {model}")
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Benchmark failed for model {model}")
            
    print("\nAll models processed.")

if __name__ == "__main__":
    main()
