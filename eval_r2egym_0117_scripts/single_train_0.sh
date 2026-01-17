#!/bin/bash
set -x

# 检查参数
if [ -z "$1" ]; then
    echo "Usage: $0 <ckpt_step>"
    echo "Example: $0 500"
    exit 1
fi

CKPT_STEP=$1

MODEL_PATH="ubowang/fim_qwen25_coder_7b_ins_0116-ckpt_${CKPT_STEP}"

DATA_PATH="/data2/yubo/datasets/R2EGym-Data/R2EGym-SFT-Trajectories.jsonl"

OUTPUT_DIR="/data2/yubo/sft_ckpts/fim_qwen25_coder_7b_ins_0116-ckpt_${CKPT_STEP}_r2egym_sft_0117"

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

export CUDA_VISIBLE_DEVICES=0,1,2,3

cd ../ms-swift

torchrun \
    --nproc_per_node 4 \
    --standalone \
    swift/cli/sft.py\
    --use_hf True \
    \
    --model $MODEL_PATH \
    --train_type full \
    --torch_dtype bfloat16 \
    \
    --dataset $DATA_PATH \
    --split_dataset_ratio 0 \
    --dataset_num_proc 8 \
    --streaming False \
    --strict False \
    --deepspeed zero3 \
    --remove_unused_columns False \
    --dataloader_num_workers 8 \
    \
    --truncation_strategy delete \
    \
    --output_dir $OUTPUT_DIR \
    --gradient_checkpointing True \
    --per_device_train_batch_size 1 \
    --weight_decay 0.05 \
    --learning_rate 1e-5 \
    --lr_scheduler_type "cosine" \
    --report_to none \
    --logging_first_step True \
    --logging_steps 1 \
    \
    --num_train_epochs 2 \
    --gradient_accumulation_steps 2 \
    --save_strategy "steps" \
    --save_steps 400 \
    --save_only_model True \
    --warmup_ratio 0.05 \
    --ddp_backend "nccl" \
    \
    --freeze_llm False \
    --freeze_vit False \
    --freeze_aligner False \
    --attn_impl flash_attn