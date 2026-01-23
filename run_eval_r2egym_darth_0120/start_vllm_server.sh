# Start VLLM server with tensor parallelism across 8 GPUs

export CUDA_VISIBLE_DEVICES=0,1

huggingface-cli download SWE-bench/SWE-agent-LM-7B --local-dir /data/yubowang/models/SWE-agent-LM-7B

SERVED_NAME="SWE-agent-LM-7B"

export MAX_CONTEXT_LEN=131072
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm serve /data/yubowang/models/SWE-agent-LM-7B \
    --port 9002 \
    --served-model-name "${SERVED_NAME}" \
    --tensor-parallel-size 1 \
    --max-model-len $MAX_CONTEXT_LEN \
    --hf-overrides '{"max_position_embeddings": '$MAX_CONTEXT_LEN'}' \
    --enable_prefix_caching





