MODELS=(
  "google/gemini-2.0-flash-001"
  "google/gemini-2.5-flash"
  "google/gemini-3-flash-preview"
  "meta-llama/llama-3.3-70b-instruct"
  "nousresearch/hermes-4-405b"
  "openai/gpt-4.1-mini"
  "openai/gpt-4o-mini"
  "openai/gpt-4o"
  "qwen/qwen3-235b-a22b-2507"
  "z-ai/glm-4-32b"
  "x-ai/grok-4.1-fast"
)

# JOIN all models as space-separated string
OBSERVERS="$(IFS=' '; echo "${MODELS[*]}")"
FOLDER="cross_results"

for TARGET in "${MODELS[@]}"; do
  touch "./${FOLDER}/${TARGET//\//_}.log"
  python cross_model_run.py \
    --target "$TARGET" \
    --observers $OBSERVERS \
    --types type2_headsup \
    --num_examples 50 \
    --threads 12 \
    --force_nocache \
    --results_dir ./${FOLDER}/ \
    --no_timestamp > "./${FOLDER}/${TARGET//\//_}.log" 2>&1 &
    # force no cache applicable only to type1_kth
done

wait

