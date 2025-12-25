import os
import json
import time
import logging
from openai import OpenAI, APIError, RateLimitError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("IntrospectionBenchmark")

class OpenRouterClient:
    def __init__(self, model_name="openai/gpt-4o", max_retries=5):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        # Default to OpenRouter URL if not set, but respect env var
        self.base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is not set.")
            
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        self.model_name = model_name
        self.max_retries = max_retries

    def generate(self, messages, temperature=0.0, max_tokens=None, item=None, stop=None, response_format=None, logprobs=False, top_logprobs=None):
        """
        Robust generation with retries.
        """
        retries = 0
        while retries < self.max_retries:
            try:
                kwargs = {
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": temperature,
                }
                if max_tokens:
                    kwargs["max_tokens"] = max_tokens
                if stop:
                    kwargs["stop"] = stop
                if response_format:
                    kwargs["response_format"] = response_format
                if logprobs:
                    kwargs["logprobs"] = True
                    if top_logprobs:
                        kwargs["top_logprobs"] = top_logprobs

                response = self.client.chat.completions.create(**kwargs)
                return response
            
            except RateLimitError:
                wait_time = (2 ** retries) + 1
                logger.warning(f"Rate limit hit. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                retries += 1
            except APIError as e:
                logger.error(f"API Error: {e}. Retrying...")
                time.sleep(2)
                retries += 1
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                raise e
        
        raise Exception(f"Failed to generate after {self.max_retries} retries.")

class CacheClient:
    def __init__(self, cache_file_path):
        self.cache = {}
        self.cache_file_path = cache_file_path
        self._load_cache()

    def _load_cache(self):
        if not os.path.exists(self.cache_file_path):
             raise FileNotFoundError(f"Cache file not found: {self.cache_file_path}")
        
        with open(self.cache_file_path, 'r') as f:
            data = json.load(f)
            # Index by ID
            for item in data:
                if 'id' in item:
                    self.cache[item['id']] = item
    
    def generate(self, messages, item=None, temperature=0.0, max_tokens=None, stop=None, response_format=None, logprobs=False, top_logprobs=None):
        if item is None:
             raise ValueError("CacheClient requires 'item' to be passed to generate().")
        
        item_id = item.get('id')
        if item_id is None:
             raise ValueError("Item passed to CacheClient must have an 'id'.")
        
        if item_id not in self.cache:
             raise ValueError(f"Item ID {item_id} not found in cache file {self.cache_file_path}.")

        cached_item = self.cache[item_id]
        
        # We expect 'raw_target_response' to be in the cached item.
        # If the task stores it as a string or structure, we wrap it in a mock response object.
        content = cached_item.get('raw_target_response', "")
        
        # Handle case where content might be structured (e.g. list/dict) -> convert to string if needed
        # But generally we want to return what would allow the task to proceed.
        # The tasks usually do `resp.choices[0].message.content`.
        
        if not isinstance(content, str):
            content = json.dumps(content) # Serialize back if it was stored as object

        # Create a mock response object structure
        class MockMessage:
            def __init__(self, content):
                self.content = content
        
        class MockChoice:
            def __init__(self, message):
                self.message = message
        
        class MockResponse:
            def __init__(self, choices):
                self.choices = choices
        
        return MockResponse([MockChoice(MockMessage(content))])

def save_result(output_dir, task_name, result_data):
    """
    Saves validation results to a JSON file.
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"{task_name}.json")
    
    # If file exists, append to list (or load and append) if it's a list structure
    # But usually benchmarks overwrite or we want a unique file per run.
    # Here we will just write/overwrite for the specific task run.
    
    with open(filepath, "w") as f:
        json.dump(result_data, f, indent=2)
    logger.info(f"Saved results to {filepath}")

def extract_json_from_response(response_text):
    """
    Robustly extracts the LAST valid JSON object from a string.
    Handles markdown code blocks and reasoning text.
    Returns the parsed dict/list or None if failed.
    """
    if not response_text:
        return None
        
    text = response_text.strip()
    
    # 1. Try to find markdown code blocks. 
    # If multiple blocks exist, we assume the answer is in the LAST one (reasoning might be in earlier blocks or text).
    import re
    code_block_pattern = r"```(?:json)?\s*(.*?)\s*```"
    matches = list(re.finditer(code_block_pattern, text, re.DOTALL))
    if matches:
        # Try parsing blocks from last to first
        for match in reversed(matches):
            block_content = match.group(1).strip()
            try:
                return json.loads(block_content)
            except:
                continue
    
    # 2. If no valid code block, try to find JSON objects in the raw text.
    # We look for substrings starting with '{' or '['.
    # We will try to find the maximal outer brackets/braces.
    
    # Simple heuristic: Find first '{' and last '}'. Try to parse.
    # If that fails, it might be due to extra text inside? Unlikely for valid JSON usage.
    # But often the model outputs: "Reasoning... { 'answer': 1 } ... more text".
    # User said "extract that json bit at the end".
    
    # Let's try to find potential JSON start characters from right to left?
    # Or just find all potential JSON-like substrings.
    
    candidates = []
    
    # Find all start indices of '{'
    brace_starts = [m.start() for m in re.finditer(r"\{", text)]
    # Find all end indices of '}'
    brace_ends = [m.start() for m in re.finditer(r"\}", text)]
    
    if brace_starts and brace_ends:
        last_end = brace_ends[-1]
        # Try to pair the last '}' with each preceding '{'
        for start in reversed(brace_starts):
            if start < last_end:
                candidate = text[start : last_end+1]
                try:
                    return json.loads(candidate)
                except:
                    continue
                    
    # Same for []
    bracket_starts = [m.start() for m in re.finditer(r"\[", text)]
    bracket_ends = [m.start() for m in re.finditer(r"\]", text)]
    
    if bracket_starts and bracket_ends:
        last_end = bracket_ends[-1]
        for start in reversed(bracket_starts):
            if start < last_end:
                candidate = text[start : last_end+1]
                try:
                    return json.loads(candidate)
                except:
                    continue

    return None
