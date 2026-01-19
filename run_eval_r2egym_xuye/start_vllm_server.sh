
export CUDA_VISIBLE_DEVICES=0
SERVED_NAME="ori_qwen25_coder_7b_ins_r2egym_sft"
MODEL_NAME="ubowang/ori_qwen25_coder_7b_ins_r2egym_sft_0108-ckpt_808"

export MAX_CONTEXT_LEN=65536
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm serve "${MODEL_NAME}" \
    --port 9002 \
    --served-model-name "${SERVED_NAME}" \
    --tensor-parallel-size 1 \
    --max-model-len $MAX_CONTEXT_LEN \
    --hf-overrides '{"max_position_embeddings": '$MAX_CONTEXT_LEN'}' \
    --enable_prefix_caching





