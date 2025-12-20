# Introspection Benchmark

A scientific benchmark for evaluating LLM introspection capabilities.

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Environment Variables**:
    Create a `.env` file:
    ```bash
    OPENROUTER_API_KEY=sk-or-your-key-here
    OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
    ```

## 1. Data Generation

Generate the synthetic datasets locally. These are saved to `data/` and will be committed to the repo.
ALREADY DONE, NO NEED TO RUN THIS
```bash
python generation_pipeline/gen_type1.py
python generation_pipeline/gen_type2.py
python generation_pipeline/gen_type3.py
```

## 2. Run Benchmark

Run the benchmark tasks. The benchmark loads the datasets from the local `data/` folder.

**Run with default model (GPT-4o Mini):**
```bash
python -m benchmark.run --task all
```


DO THIS ONE OVER ALL THE MODELS WE WANT TO TEST. TRY TO USE THE LITE VERSIONS OF MODELS WHEREVER POSSIBLE

**Run with Nemotron 9B (via OpenRouter):**
```bash
python -m benchmark.run --task all --model nvidia/nemotron-nano-9b-v2
```

## Results

Results are saved to `results/<model_name>/<task_name>.json`.
