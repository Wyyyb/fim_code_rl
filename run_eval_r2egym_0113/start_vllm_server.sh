# Start VLLM server with tensor parallelism across 8 GPUs

export CUDA_VISIBLE_DEVICES=4,5,6,7

export MAX_CONTEXT_LEN=65536
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm serve /data/yubowang/fim_sft_ckpts/ori_qwen25_coder_7b_ins_r2egym_sft_0108-ckpt_808 \
    --tensor-parallel-size 4 \
    --max-model-len $MAX_CONTEXT_LEN \
    --hf-overrides '{"max_position_embeddings": '$MAX_CONTEXT_LEN'}' \
    --enable_prefix_caching





