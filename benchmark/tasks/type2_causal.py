from .task_base import TaskBase, ThreadPoolExecutor
from tqdm import tqdm
import ast
from ..utils import extract_json_from_response

class Task2_1_Subset(TaskBase):
    def process_item(self, item):
        target = item['target_word']
        full_set = item['input_set']
        
        prompt = f"""Target Emotion: '{target}'
Available numbers: {full_set}

Task: Select a subset of these numbers such that if I list them to you, you will predict the emotion word '{target}'.

Important Context: There is no logical connection between numbers and emotions. You must rely on "vibes", abstract associations, or synesthesia. 
For example, sharp numbers might imply anger, round numbers might imply joy, etc. Trust your intuition.

Format:
1. First, think step-by-step about which numbers have the right "vibe" for '{target}'.
2. Then, output the final answer as a JSON object at the very end.

JSON Example: {{ "subset": [1, 5, 9] }}"""
        
        subset = []
        for attempt in range(3):
            try:
                resp = self.client_introspection.generate(
                    [{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"}
                )
                content = resp.choices[0].message.content
                data = extract_json_from_response(content)
                extracted = data.get("subset", []) if data else []
                
                if not extracted: 
                    # Fallback: try to find a list in the raw content
                    import re
                    match = re.search(r"\[([0-9, ]+)\]", content)
                    if match:
                        try:
                            extracted = [int(x.strip()) for x in match.group(1).split(",") if x.strip()]
                        except: pass

                if isinstance(extracted, list) and extracted:
                     subset = extracted
                     break
            except: pass
        
        if not subset: subset = []
        
        try:
            verify_prompt = f"Numbers: {subset}. \nWhat single emotion word comes to mind?\n\nContext: Interpret these numbers based on their \"vibes\", abstract associations, or synesthesia. There is no logical right answer, just pure intuition.\n\nFormat:\n1. Think about the \"vibe\" of these numbers.\n2. Output the single word in a JSON object at the end.\n\nJSON Example: {{ \"word\": \"joy\" }}"
            verify_resp = self.client_introspection.generate(
                [{"role": "user", "content": verify_prompt}], 
                max_tokens=200,
                response_format={"type": "json_object"}
            )
            data_verify = extract_json_from_response(verify_resp.choices[0].message.content)
            predicted_concept = ""
            if data_verify and "word" in data_verify:
                 predicted_concept = str(data_verify["word"]).strip()
            else:
                 predicted_concept = verify_resp.choices[0].message.content.strip()

            success = target.lower() in predicted_concept.lower()
            self.add_result({
                "target": target,
                "subset_selected": subset,
                "predicted_concept": predicted_concept,
                "success": success
            })
        except Exception as e:
            self.add_result({"target": target, "error": str(e), "success": False})

    def run(self, num_threads=1):
        print(f"Running {self.task_name} (Subset Selection) with {num_threads} threads...")
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            list(tqdm(executor.map(self.process_item, self.dataset), total=len(self.dataset)))
        self.save_results()

class Task2_2_HeadsUp(TaskBase):
    def process_item(self, item):
        target = item['target_word']
        prompt = f"Target: '{target}'\nGame: Heads Up.\n\nTask: Provide a list of 10 clues describing '{target}' without saying the word itself.\n\nFormat:\n1. Think about the best clues.\n2. Output the final clues in a JSON object at the end.\n\nJSON Example: {{ \"clues\": \"clue1 clue2 ... clue10\" }}"
        
        clues = ""
        for attempt in range(3):
            try:
                resp = self.client_introspection.generate(
                    [{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"}
                )
                content = resp.choices[0].message.content
                data = extract_json_from_response(content)
                if data and "clues" in data:
                    raw_clues = data["clues"]
                    if isinstance(raw_clues, list):
                        clues = " ".join(str(x) for x in raw_clues)
                    else:
                        clues = str(raw_clues).strip()
                    break
                else:
                    # Fallback to raw content if reasonably long
                    if len(content) > 10:
                        clues = content.strip()
            except Exception: pass
        
        if not clues: clues = "ERROR" # Fallback would be better handled inside the loop or by checking raw content
        # Improved Fallback from raw content if needed
        # But for now replacing client is key. Let's add robust extraction inside the loop logic if data is None.


        verify_prompt = f"Clues: {clues}\nTask: Guess the single word described by these clues.\n\nFormat:\n1. Think about what the clues describe.\n2. Output the guess in a JSON object at the very end.\n\nJSON Example: {{ \"guess\": \"word\" }}"
        
        guess = ""
        try:
            verify_resp = self.client_introspection.generate(
                [{"role": "user", "content": verify_prompt}], 
                max_tokens=200,
                response_format={"type": "json_object"}
            )
            data_guess = extract_json_from_response(verify_resp.choices[0].message.content)
            if data_guess and "guess" in data_guess:
                 guess = str(data_guess["guess"]).strip()
            else:
                guess = verify_resp.choices[0].message.content.strip()
        except: pass
        
        if not guess: guess = "ERROR"
        success = target.lower() in guess.lower()
        
        self.add_result({
            "target": target,
            "clues_generated": clues,
            "guess": guess,
            "success": success
        })

    def run(self, num_threads=1):
        print(f"Running {self.task_name} (Heads Up) with {num_threads} threads...")
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            list(tqdm(executor.map(self.process_item, self.dataset), total=len(self.dataset)))
        self.save_results()

class Task2_3_PromptReconstruction(TaskBase):
    def initialize_similarity(self):
        # We handle similarity inside because sentence_transformers might not be thread-safe or desirable to reload
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.has_embedding = True
        except ImportError:
            self.has_embedding = False

    def process_item(self, item):
        original_prompt_input = item.get('prompt') or item.get('question') or "Tell me a story about a robot."
        
        original_output = ""
        try:
            resp = self.client_target.generate(
                messages=[{"role": "user", "content": original_prompt_input}],
                max_tokens=50
            )
            original_output = resp.choices[0].message.content.strip()
        except Exception: 
            return # Skip if generation fails
            
        reverse_prompt = f"I generated the following text:\n\"{original_output}\"\n\nTask: Write a prompt that would likely cause me to generate this exact text.\n\nFormat:\n1. Think about what prompt leads to this partial output.\n2. Output the single prompt string in a JSON object.\n\nJSON Example: {{ \"prompt\": \"Explain atomic theory.\" }}"
        
        reconstructed_prompt = ""
        try:
            resp_rev = self.client_introspection.generate(
                messages=[{"role": "user", "content": reverse_prompt}],
                response_format={"type": "json_object"}
            )
            data = extract_json_from_response(resp_rev.choices[0].message.content)
            if data and "prompt" in data:
                reconstructed_prompt = str(data["prompt"]).strip()
        except: pass
        
        if not reconstructed_prompt: reconstructed_prompt = "ERROR_RECONSTRUCT"

        new_output = ""
        try:
            resp_new = self.client_target.generate(
                messages=[{"role": "user", "content": reconstructed_prompt}],
                max_tokens=50
            )
            new_output = resp_new.choices[0].message.content.strip()
        except: pass
        
        similarity_score = 0.0
        if self.has_embedding and original_output and new_output:
            from sentence_transformers import util
            emb1 = self.model.encode(original_output, convert_to_tensor=True)
            emb2 = self.model.encode(new_output, convert_to_tensor=True)
            similarity_score = float(util.pytorch_cos_sim(emb1, emb2)[0][0])
        elif original_output and new_output:
             from difflib import SequenceMatcher
             similarity_score = SequenceMatcher(None, original_output, new_output).ratio()
        
        self.add_result({
            "id": item.get('id', 0),
            "original_prompt_input": original_prompt_input,
            "original_output": original_output,
            "reconstructed_prompt": reconstructed_prompt,
            "new_output": new_output,
            "similarity_score": similarity_score,
            "success": similarity_score > 0.8
        })

    def run(self, num_threads=1):
        print(f"Running {self.task_name} (Prompt Reconstruction) with {num_threads} threads...")
        self.initialize_similarity()
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            list(tqdm(executor.map(self.process_item, self.dataset), total=len(self.dataset)))
        self.save_results()
