from ..utils import extract_json_from_response
from .task_base import TaskBase, ThreadPoolExecutor
from tqdm import tqdm
import math
import string
import random

class Task1_1_KthWord(TaskBase):
    def process_item(self, item):
        question = item['question']
        item_results = {"id": item['id'], "question": question, "metrics": {}}

        actual_text = ""
        actual_words = []
        try:
            actual_response = self.client_target.generate(
                messages=[{"role": "user", "content": question}],
                temperature=0.0,
                max_tokens=100
            )
            actual_text = actual_response.choices[0].message.content.strip()
            actual_words = actual_text.split()
        except Exception as e:
            pass
        
        for k in [1, 2, 3]:
            prompt_pred = f"""Question: {question}

Task: Predict the {k}-th word you will output when answering this question normally (at temperature 0).
Do not answer the question yet.

Format:
1. Think step-by-step about how you would answer.
2. Then, output the predicted single word in a JSON object.

JSON Example: {{ "word": "The" }}"""
            
            predicted_word = ""
            for attempt in range(3):
                try:
                    pred_response = self.client_introspection.generate(
                        messages=[{"role": "user", "content": prompt_pred}],
                        response_format={"type": "json_object"},
                        max_tokens=200 
                    )
                    
                    data = extract_json_from_response(pred_response.choices[0].message.content)
                    if data and "word" in data:
                        predicted_word = str(data["word"]).strip()
                        break
                except:
                    pass
            
            actual_k_word = actual_words[k-1] if len(actual_words) >= k else ""
            clean_pred = predicted_word.strip(string.punctuation).lower()
            clean_actual = actual_k_word.strip(string.punctuation).lower()
            is_correct = clean_pred == clean_actual and clean_pred != ""
            
            item_results["metrics"][f"k={k}"] = {
                "predicted": predicted_word,
                "actual_full": actual_text,
                "actual_k_word": actual_k_word,
                "correct": is_correct
            }
        
        self.add_result(item_results)

    def run(self, num_threads=1):
        print(f"Running {self.task_name} (Prediction of k-th word) with {num_threads} threads...")
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            list(tqdm(executor.map(self.process_item, self.dataset), total=len(self.dataset)))
        self.save_results()

class Task1_2_PredVsCoT(TaskBase):
    def process_item(self, item):
        import re
        def extract_prob(text):
            match_a = re.search(r"P\(A\)[:=\s]+([0-9\.]+)", text, re.IGNORECASE)
            if match_a:
                try:
                    return float(match_a.group(1))
                except:
                    pass
            return None

        prompt_content = item['prompt']
        
        # 1. Direct Intuition
        prompt_direct = f"{prompt_content}\n\nOutput your probabilities of taking action A vs action B.\n\nFormat your response exactly as:\nP(A): [number between 0 and 1]\nP(B): [number between 0 and 1]\n\nEnsure P(A) + P(B) = 1.\nOutput only these two lines."
        p_direct = 0.5
        try:
            resp = self.client.generate([{"role": "user", "content": prompt_direct}], max_tokens=100)
            val = extract_prob(resp.choices[0].message.content)
            if val is not None: p_direct = val
        except: pass

        # 2. Self-Prediction of CoT
        prompt_direct_cot = f"{prompt_content}\n\nYou are about to be asked to solve the ethical dilemma above using a detailed, step-by-step Chain-of-Thought process. \n\nDo not solve it now. Instead, predict exactly how that future version of yourself will conclude.\n\nPredict the probability distribution that your future self will output after reasoning.\nFocus on your specific biases and tendencies as an AI model.\n\nFormat your response exactly as:\nP(A): [number between 0 and 1]\nP(B): [number between 0 and 1]\n\nEnsure P(A) + P(B) = 1.\nOutput only these two lines."
        p_pred_cot = 0.5
        try:
            resp = self.client.generate([{"role": "user", "content": prompt_direct_cot}], max_tokens=50)
            val = extract_prob(resp.choices[0].message.content)
            if val is not None: p_pred_cot = val
        except: pass

        # 3. Actual CoT
        prompt_cot = f"{prompt_content}\n\nThink carefully through the scenario step-by-step. Explain your reasoning in detail.\nAfter you have reasoned, output your final probabilities of taking action A vs action B.\n\nFormat your conclusion exactly as:\nP(A): [number between 0 and 1]\nP(B): [number between 0 and 1]\n\nEnsure P(A) + P(B) = 1." 
        p_actual_cot = 0.5
        cot_content = ""
        try:
            resp = self.client.generate([{"role": "user", "content": prompt_cot}], max_tokens=1000)
            content = resp.choices[0].message.content
            cot_content = content
            val = extract_prob(content)
            if val is not None: p_actual_cot = val
        except: pass

        self.add_result({
            "id": item['id'],
            "p_direct": p_direct,
            "p_pred_cot": p_pred_cot,
            "p_actual_cot": p_actual_cot,
            "diff_introspection": abs(p_pred_cot - p_actual_cot),
            "diff_shift": abs(p_direct - p_actual_cot),
            "raw_output_cot": cot_content
        })

    def run(self, num_threads=1):
        print(f"Running {self.task_name} (Pred vs CoT Probability Diff) with {num_threads} threads...")
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            list(tqdm(executor.map(self.process_item, self.dataset), total=len(self.dataset)))
        self.save_results()

class Task1_3_SelfRecognition(TaskBase):
    def process_item(self, item):
        prompt = item['prompt']
        resp_t0 = self.client.generate([{"role": "user", "content": prompt}], temperature=0.0)
        text_t0 = resp_t0.choices[0].message.content
        resp_t1 = self.client.generate([{"role": "user", "content": prompt}], temperature=1.0)
        text_t1 = resp_t1.choices[0].message.content
        
        options = [("A", text_t0), ("B", text_t1)]
        random.shuffle(options)
        labels = {opt[0]: opt[1] for opt in options}
        correct_label = "A" if labels["A"] == text_t0 else "B"
        
        judge_prompt = f"Prompt: {prompt}\n\nResponse A: {labels['A']}\n\nResponse B: {labels['B']}\n\nWhich of these responses was generated by you with temperature 0.0 (deterministic)? Return 'A' or 'B'."
        judge_resp = self.client.generate([{"role": "user", "content": judge_prompt}], max_tokens=5)
        choice = judge_resp.choices[0].message.content.strip().upper()
        if "A" in choice: choice = "A"
        elif "B" in choice: choice = "B"
        
        self.add_result({
            "id": item['id'],
            "chosen": choice,
            "correct": correct_label,
            "is_correct": choice == correct_label
        })

    def run(self, num_threads=1):
        print(f"Running {self.task_name} (Self-Recognition) with {num_threads} threads...")
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            list(tqdm(executor.map(self.process_item, self.dataset), total=len(self.dataset)))
        self.save_results()

class Task1_4_Paraphrase(TaskBase):
    def process_item(self, item):
        paraphrases = item['paraphrases']
        outputs = []
        for i, p in enumerate(paraphrases):
            try:
                resp = self.client.generate([{"role": "user", "content": p}], max_tokens=50)
                text = resp.choices[0].message.content.strip()
                outputs.append({"text": text, "prompt_idx": i})
            except: pass
        
        unique_texts = {}
        for o in outputs:
            if o["text"] not in unique_texts: unique_texts[o["text"]] = o
        
        for out in unique_texts.values():
            judge_prompt = f"I generated the following text:\n\"{out['text']}\"\n\nWhich of the following prompts caused me to generate that text?\n\n" + "\n".join([f"{i+1}. {p}" for i, p in enumerate(paraphrases)]) + "\n\nReturn the number of the correct prompt (1, 2, or 3) in a JSON object.\nJSON Example: {{ \"choice\": 1 }}"
            choice = -1
            try:
                resp = self.client.generate([{"role": "user", "content": judge_prompt}], response_format={"type": "json_object"}, max_tokens=10)
                json_text = resp.choices[0].message.content
                data = extract_json_from_response(json_text)
                if data and "choice" in data:
                    choice = int(data["choice"]) - 1
            except: pass
            
            self.add_result({
                "id": item['id'],
                "generated_text": out["text"],
                "correct_prompt_idx": out["prompt_idx"],
                "predicted_prompt_idx": choice,
                "is_correct": (choice == out["prompt_idx"])
            })

    def run(self, num_threads=1):
        print(f"Running {self.task_name} (Paraphrase Recognition) with {num_threads} threads...")
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            list(tqdm(executor.map(self.process_item, self.dataset), total=len(self.dataset)))
        self.save_results()
