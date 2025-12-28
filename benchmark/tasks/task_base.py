from threading import Lock
from concurrent.futures import ThreadPoolExecutor
from abc import ABC, abstractmethod
from datasets import load_dataset
import logging
import os
from tqdm import tqdm

logger = logging.getLogger("IntrospectionBenchmark")

class TaskBase(ABC):
    def __init__(self, task_name, dataset_name, dataset_split, client_target, client_introspection=None, output_dir="results", max_tokens=None):
        self.task_name = task_name
        self.client_target = client_target
        self.client_introspection = client_introspection if client_introspection else client_target
        self.output_dir = output_dir
        self.dataset_split = dataset_split
        self.max_tokens = max_tokens
        self.dataset = self.load_data(dataset_name, dataset_split)
        self.results = []
        self.results_lock = Lock()

    def load_data(self, dataset_name, split):
        logger.info(f"Loading dataset from local file: data/{split}.json")
        file_path = f"data/{split}.json"
        if not os.path.exists(file_path):
             raise FileNotFoundError(f"Data file not found: {file_path}. Did you run the generation scripts?")
        
        return load_dataset("json", data_files=file_path)["train"]

    @abstractmethod
    def run(self, num_threads=1):
        """
        Executes the benchmark task on the dataset.
        """
        pass

    def add_result(self, result):
        with self.results_lock:
            self.results.append(result)

    def save_results(self):
        from benchmark.utils import save_result
        save_result(self.output_dir, self.task_name, self.results)
