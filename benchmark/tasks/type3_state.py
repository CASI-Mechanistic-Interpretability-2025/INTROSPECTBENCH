from .task_base import TaskBase, ThreadPoolExecutor
from tqdm import tqdm
import math
import json
import logging
from ..utils import extract_json_from_response

logger = logging.getLogger("IntrospectionBenchmark")

class Task3_1_ProbTargeting(TaskBase):
    def process_item(self, item):
        category = item['category']
        targets = item['target_probs']
        
        item_results = {"category": category, "trials": []}
        
        for p_target in targets:
            prompt_req = f"Construct a prompt about '{category}' where the probability of your top predicted token for the response is exactly {p_target}.\nReturn a JSON object with a single key 'prompt' containing the prompt string."
            resp = self.client_target.generate(
                [{"role": "user", "content": prompt_req}],
                response_format={"type": "json_object"}
            )
            try:
                data = extract_json_from_response(resp.choices[0].message.content)
                generated_prompt = data.get("prompt", "") if data else resp.choices[0].message.content.strip()
            except:
                generated_prompt = resp.choices[0].message.content.strip()
            
            try:
                eval_resp = self.client_target.generate(
                    [{"role": "user", "content": generated_prompt}],
                    max_tokens=1,
                    logprobs=True,
                    top_logprobs=1
                )
                
                if eval_resp.choices[0].logprobs and eval_resp.choices[0].logprobs.content:
                    token_info = eval_resp.choices[0].logprobs.content[0]
                    logprob = token_info.logprob
                    p_actual = math.exp(logprob)
                    top_token = token_info.token
                else:
                    p_actual = 0.0
                    top_token = "ERROR"

                mse = (p_target - p_actual) ** 2
                item_results["trials"].append({
                    "target_p": p_target,
                    "generated_prompt": generated_prompt,
                    "actual_p": p_actual,
                    "top_token": top_token,
                    "mse": mse
                })
            except Exception as e:
                 item_results["trials"].append({
                    "target_p": p_target,
                    "error": str(e)
                })
        
        self.add_result(item_results)

    def run(self, num_threads=1):
        print(f"Running {self.task_name} (Probability Targeting) with {num_threads} threads...")
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            list(tqdm(executor.map(self.process_item, self.dataset), total=len(self.dataset)))
        self.save_results()
