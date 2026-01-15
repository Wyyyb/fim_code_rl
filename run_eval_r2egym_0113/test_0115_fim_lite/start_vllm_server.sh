# Start VLLM server with tensor parallelism across 8 GPUs

export CUDA_VISIBLE_DEVICES=7
SERVED_NAME="fim_qwen25_coder_7b_ins_0105_r2egym_sft_0108-ckpt_808-lite"

export MAX_CONTEXT_LEN=65536
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm serve /data/yubowang/fim_sft_ckpts/fim_qwen25_coder_7b_ins_0105_r2egym_sft_0108-ckpt_808 \
    --port 9013 \
    --served-model-name "${SERVED_NAME}" \
    --tensor-parallel-size 1 \
    --max-model-len $MAX_CONTEXT_LEN \
    --hf-overrides '{"max_position_embeddings": '$MAX_CONTEXT_LEN'}' \
    --enable_prefix_caching





