import argparse
import os
import sys
import time
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from transformers import BitsAndBytesConfig
# Imports assuming correct package structure or running from introspectionbench dir
from benchmark.utils import OpenRouterClient, save_result
from benchmark.tasks.type1_self_pred import Task1_1_KthWord, Task1_2_PredVsCoT, Task1_3_SelfRecognition
from benchmark.tasks.type2_causal import Task2_1_Subset, Task2_2_HeadsUp, Task2_3_PromptReconstruction
from benchmark.tasks.type3_state import Task3_1_ProbTargeting

class LocalHFClient:
    def __init__(self, model_path, is_adapter=False, base_model_name=None, quantize_8bit=False, device="auto", force_tie_weights=False):
        # Handle cases where user points to a file instead of dir
        if os.path.isfile(model_path):
            print(f"Provided path is a file, using parent directory: {os.path.dirname(model_path)}")
            model_path = os.path.dirname(model_path)
        
        # Resolve to abspath if it exists locally
        if os.path.exists(model_path):
            model_path = os.path.abspath(model_path)
            
        print(f"Loading local model from {model_path}...")
        self.device = device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        
        print(f"Using device: {self.device}")
        if self.device == "cuda":
            print(f"  CUDA Device Count: {torch.cuda.device_count()}")
            print(f"  Current Device: {torch.cuda.current_device()}")
            print(f"  Device Name: {torch.cuda.get_device_name(0)}")
        elif self.device == "cpu":
             print("  WARNING: Running on CPU. This will be very slow.")

        # Load Tokenizer
        tokenizer_path = base_model_name if is_adapter else model_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, legacy=False, trust_remote_code=True)
        
        # Load Model
        if quantize_8bit:
            print("Quantizing model to 8-bit...")
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map=self.device,
                trust_remote_code=True,
                quantization_config=bnb_config
            )
        elif is_adapter:
            print(f"Loading base model: {base_model_name}")
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name, 
                torch_dtype=torch.float16,
                device_map=self.device,
                trust_remote_code=True
            )
            print(f"Loading adapter: {model_path}")
            self.model = PeftModel.from_pretrained(base_model, model_path)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                torch_dtype=torch.float16,
                device_map=self.device,
                trust_remote_code=True
            )
        
        # Weight Tying Logic
        if force_tie_weights:
            print("Forcing weight tying (lm_head = embed_tokens)...")
            try:
                self.model.lm_head.weight = self.model.model.embed_tokens.weight
                self.model.tie_weights()
            except Exception as e:
                print(f"Failed to force tie weights: {e}")
        else:
            self.model.tie_weights()
        
        self.model.eval()

    def generate(self, messages, temperature=0.7, max_tokens=100, response_format=None, logprobs=False, top_logprobs=None, **kwargs):
        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        do_sample = temperature > 0
        
        # Prepare generation args
        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": do_sample,
            "temperature": temperature if do_sample else None,
            "pad_token_id": self.tokenizer.eos_token_id,
            **kwargs
        }

        # If logprobs requested, we need scores
        if logprobs:
            gen_kwargs["output_scores"] = True
            gen_kwargs["return_dict_in_generate"] = True

        start_time = time.time()
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        end_time = time.time()
        
        # Calculate speed
        duration = end_time - start_time
        num_generated_tokens = 0
        if isinstance(outputs, torch.Tensor):
             num_generated_tokens = outputs.shape[1] - inputs.input_ids.shape[1]
        elif hasattr(outputs, 'sequences'):
             num_generated_tokens = outputs.sequences.shape[1] - inputs.input_ids.shape[1]
            
        if duration > 0:
            speed = num_generated_tokens / duration
            print(f"Generation took {duration:.2f}s for {num_generated_tokens} tokens ({speed:.2f} t/s)")
        
        if logprobs:
            # outputs is a ModelOutput object
            generated_sequences = outputs.sequences
            scores = outputs.scores # Tuple of tensors (one per step)
            
            # Extract generated ID part
            input_len = inputs.input_ids.shape[1]
            generated_ids = generated_sequences[0][input_len:]
            
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # Process Logprobs
            # OpenAI format: response.choices[0].logprobs.content -> list of TokenInfo
            # TokenInfo has .token (str) and .logprob (float)
            
            token_infos = []
            
            # scores is tuple of len(generated_ids)
            # each element is tensor [batch_size, vocab_size]
            for i, token_id in enumerate(generated_ids):
                if i >= len(scores): break # Should not happen usually
                
                step_scores = scores[i][0] # Batch 0
                step_logprobs = torch.nn.functional.log_softmax(step_scores, dim=-1)
                token_logprob = step_logprobs[token_id].item()
                token_str = self.tokenizer.decode([token_id])
                
                class MockTokenInfo:
                    def __init__(self, token, logprob):
                        self.token = token
                        self.logprob = logprob
                
                token_infos.append(MockTokenInfo(token_str, token_logprob))

            class MockLogprobs:
                def __init__(self, content):
                    self.content = content
            
            mock_logprobs_obj = MockLogprobs(token_infos)

        else:
            # outputs is just the tensor if return_dict_in_generate=False (default behavior of simple generate)
            # BUT we kept return_dict_in_generate=False above for the else case to imply 'outputs' is tensor? 
            # Actually huggingface output depends on config. Let's handle standard tensor output or ModelOutput
            if isinstance(outputs, torch.Tensor):
                generated_ids = outputs[0][inputs.input_ids.shape[1]:]
            else:
                generated_ids = outputs.sequences[0][inputs.input_ids.shape[1]:]
                
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            mock_logprobs_obj = None
        
        # Mock response object to match OpenRouter/OpenAI API structure
        class MockMessage:
            def __init__(self, content):
                self.content = content
        
        class MockChoice:
            def __init__(self, message, logprobs=None):
                self.message = message
                self.logprobs = logprobs
        
        class MockResponse:
            def __init__(self, choices):
                self.choices = choices
                
        return MockResponse([MockChoice(MockMessage(generated_text), logprobs=mock_logprobs_obj)])

def main():
    parser = argparse.ArgumentParser(description="Run Introspection Benchmark (Small/Local)")
    parser.add_argument("--task", type=str, default="all", 
                        help="Task to run: type1_kth, type1_cot, type1_recog, type2_subset, type2_headsup, type2_deception, type3_prob, or 'all'")
    parser.add_argument("--model_path", type=str, required=True, help="Path to local model OR OpenRouter model name")
    parser.add_argument("--is_local", action="store_true", help="Flag to indicate if using a local HF model")
    parser.add_argument("--is_adapter", action="store_true", help="Flag to indicate if the local path is a LoRA adapter")
    parser.add_argument("--base_model", type=str, default=None, help="Base model name if loading adapter")
    parser.add_argument("--limit", type=int, default=200, help="Number of items to run per task")
    parser.add_argument("--output_dir", type=str, default="results_experiments", help="Directory to save results")
    parser.add_argument("--max_tokens", type=int, default=None, help="Max tokens to generate")
    parser.add_argument("--quantize_8bit", action="store_true", help="Flag to indicate if the local path is a LoRA adapter")
    parser.add_argument("--force_tie_weights", action="store_true", help="Force tying variables")
    # Introspection Model Arguments
    parser.add_argument("--introspection_model_path", type=str, default=None, help="Path to introspection model (Guesser). Defaults to target model if not set.")
    parser.add_argument("--introspection_is_local", action="store_true", help="Flag if introspection model is local")
    parser.add_argument("--introspection_is_adapter", action="store_true", help="Flag if introspection model is a LoRA adapter")
    parser.add_argument("--introspection_base_model", type=str, default=None, help="Base model name for introspection adapter")
    
    args = parser.parse_args()
    
    # Initialize Client
    if args.is_local:
        if args.is_adapter and not args.base_model:
            raise ValueError("--base_model must be provided when --is_adapter is set")
        client = LocalHFClient(args.model_path, is_adapter=args.is_adapter, base_model_name=args.base_model, force_tie_weights=args.force_tie_weights)
        model_id = os.path.basename(args.model_path) if args.is_local else args.model_path.replace("/", "_")
    else:
        client = OpenRouterClient(model_name=args.model_path)
        model_id = args.model_path.replace("/", "_")

    # Initialize Introspection Client
    client_introspection = None
    introspection_model_id = model_id # Default to same ID if not different

    if args.introspection_model_path:
        print(f"Initializing Introspection Client: {args.introspection_model_path}")
        if args.introspection_is_local:
             if args.introspection_is_adapter and not args.introspection_base_model:
                raise ValueError("--introspection_base_model must be provided when --introspection_is_adapter is set")
             client_introspection = LocalHFClient(
                 args.introspection_model_path, 
                 is_adapter=args.introspection_is_adapter, 
                 base_model_name=args.introspection_base_model,
                 # max_tokens arg was not in init signature but passed in original code?? 
                 # Removing max_tokens from init as it is not in the def __init__
                 quantize_8bit=args.quantize_8bit,
                 force_tie_weights=args.force_tie_weights
             )
             introspection_model_id = os.path.basename(args.introspection_model_path) if args.introspection_is_local else args.introspection_model_path.replace("/", "_")
        else:
            client_introspection = OpenRouterClient(model_name=args.introspection_model_path)
            introspection_model_id = args.introspection_model_path.replace("/", "_")
    else:
        print("Using target model for introspection (Self-Introspection).")
        client_introspection = client 
        
    # Construct combined model ID for output
    if args.introspection_model_path:
        combined_model_id = f"{model_id}_VS_{introspection_model_id}"
    else:
        combined_model_id = model_id

    # Create Date-Stamped Output Directory
    from datetime import datetime
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(args.output_dir, combined_model_id, date_str)
    os.makedirs(output_dir, exist_ok=True)
    
    # Task Registry (Similar to original run.py)
    # Using 'local' as placeholder repo_id
    repo_id_placeholder = "local"
    
    tasks = []
    
    if args.task in ["type1_kth", "all"]:
        tasks.append(Task1_1_KthWord("type1_kth_word", repo_id_placeholder, "type1_nq", client, client_introspection=client_introspection, output_dir=output_dir, max_tokens=args.max_tokens))
        
    if args.task in ["type1_cot", "all"]:
        tasks.append(Task1_2_PredVsCoT("type1_cot_forced", repo_id_placeholder, "type1_open", client, client_introspection=client_introspection, output_dir=output_dir, max_tokens=args.max_tokens))
        
    if args.task in ["type1_recog", "all"]:
        tasks.append(Task1_3_SelfRecognition("type1_self_recognition", repo_id_placeholder, "type1_open", client, client_introspection=client_introspection, output_dir=output_dir, max_tokens=args.max_tokens))
        
    if args.task in ["type2_subset", "all"]:
        tasks.append(Task2_1_Subset("type2_subset", repo_id_placeholder, "type2_subset", client, client_introspection=client_introspection, output_dir=output_dir, max_tokens=args.max_tokens))
        
    if args.task in ["type2_headsup", "all"]:
        tasks.append(Task2_2_HeadsUp("type2_headsup", repo_id_placeholder, "type2_headsup", client, client_introspection=client_introspection, output_dir=output_dir, max_tokens=args.max_tokens))
        
    if args.task in ["type2_deception", "all"]:
        tasks.append(Task2_3_PromptReconstruction("type2_deception", repo_id_placeholder, "type2_deception", client, client_introspection=client_introspection, output_dir=output_dir, max_tokens=args.max_tokens))
        
    if args.task in ["type3_prob", "all"]:
        tasks.append(Task3_1_ProbTargeting("type3_prob_targeting", repo_id_placeholder, "type3_probs", client, client_introspection=client_introspection, output_dir=output_dir, max_tokens=args.max_tokens))

    print(f"Starting benchmark run for {args.model_path} (Limit: {args.limit})...")
    print(f"Saving results to: {output_dir}")
    
    all_scores = {}
    
    for task in tasks:
        try:
            # Apply Limit
            if args.limit and len(task.dataset) > args.limit:
                print(f"Limiting task {task.task_name} from {len(task.dataset)} to {args.limit} items.")
                task.dataset = task.dataset.select(range(args.limit))
            
            task.run(num_threads=1) # Local models usually don't support multithreading well due to VRAM
            
            # Simple scoring summary logic
            task_score = None
            task_metric = ""
            
            # This relies on the result files being saved to disk and reloadable, or inspecting task.results object
            # Since task.results is populated, we can use it.
            
            if task.task_name == "type1_kth_word":
                correct = sum(1 for r in task.results if r["metrics"]["k=1"]["correct"])
                total = len(task.results)
                if total > 0:
                    task_score = (correct / total) * 100
                    task_metric = "Accuracy (k=1)"
            
            elif task.task_name == "type1_cot_forced":
                 # Score is 1 - average shift
                total_shift = sum(r["diff_introspection"] for r in task.results if "diff_introspection" in r)
                count = len(task.results)
                if count > 0:
                    avg_shift = total_shift / count
                    task_score = (1.0 - avg_shift)
                    task_metric = f"1 - AvgShift[{avg_shift:.2f}]"
            
            elif task.task_name == "type1_self_recognition":
                correct = sum(1 for r in task.results if r["is_correct"])
                total = len(task.results)
                if total > 0:
                    task_score = (correct / total) * 100
                    task_metric = "Accuracy"
                    
            elif task.task_name == "type2_subset":
                 # Parsing success logic is complex in subset task, assuming 'is_correct' or similar exists?
                 # Checking type2_causal.py... it saves 'success' boolean usually.
                 # Let's assume standard 'success' or 'correct' usage if available, else skip for now
                 pass

            if task_score is not None:
                all_scores[task.task_name] = (task_score, task_metric)
                
        except Exception as e:
            print(f"Task {task.task_name} failed: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n==================== Results Summary for {model_id} ====================")
    for name, (score, metric) in all_scores.items():
        print(f"{name}: {score:.2f} ({metric})")
    
    print("\nDetailed results saved in:")
    print(output_dir)

if __name__ == "__main__":
    main()
