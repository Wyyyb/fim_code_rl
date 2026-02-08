set -x

# conda activate YOUR_ENV

MODEL_PATH="/data/yubo/models/Qwen3-Coder-30B-A3B-Instruct"

DATA_PATH="/data/yubo/datasets/R2EGym-Data/R2EGym-SFT-Trajectories.jsonl"

OUTPUT_DIR="/data/yubo/sft_ckpts/qwen3_coder_30b_a3b_ins_r2egym_sft_0208"

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

cd ../ms-swift

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NNODES=1 \
NODE_RANK=0 \
megatron sft \
    --model $MODEL_PATH \
    --load_safetensors true \
    --save_safetensors true \
    --dataset $DATA_PATH \
    --load_from_cache_file true \
    --pipeline_model_parallel_size 1 \
    --expert_model_parallel_size 4 \
    --context_parallel_size 2 \
    --moe_permute_fusion true \
    --moe_grouped_gemm true \
    --moe_shared_expert_overlap true \
    --moe_aux_loss_coeff 1e-3 \
    --micro_batch_size 1 \
    --global_batch_size 8 \
    --packing true \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --max_epochs 2 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-5 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-6 \
    --save $OUTPUT_DIR \
    --save_interval 400 \
    --max_length 32768 \
    --num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim true \
    --no_save_rng true \
    --sequence_parallel true \
    --attention_backend flash \
    --optimizer_cpu_offload true \
    --use_precision_aware_optimizer true \
    --optimizer_offload_fraction 1