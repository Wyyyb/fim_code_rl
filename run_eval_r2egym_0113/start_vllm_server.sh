# Start VLLM server with tensor parallelism across 8 GPUs

export CUDA_VISIBLE_DEVICES=4

export MAX_CONTEXT_LEN=65536
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm serve ubowang/ori_qwen25_coder_7b_ins_r2egym_sft_0108-ckpt_808 \
    --host 0.0.0.0 \
    --port 9002 \
    --tensor-parallel-size 1 \
    --max-model-len $MAX_CONTEXT_LEN \
    --hf-overrides '{"max_position_embeddings": '$MAX_CONTEXT_LEN'}' \
    --enable_prefix_caching





