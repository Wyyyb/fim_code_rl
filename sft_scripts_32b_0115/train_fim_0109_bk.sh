set -x

# conda activate YOUR_ENV

MODEL_PATH="/data2/yubo/models/Qwen2.5-Coder-32B-Instruct"

DATA_PATH="/data2/yubo/datasets/fim_midtrain_data_0108_212k/fim_sft_data_0108.jsonl"

OUTPUT_DIR="/data2/yubo/sft_ckpts/fim_qwen25_coder_32b_ins_0118_lora_ckpt"

MERGED_DIR="/data2/yubo/sft_ckpts/fim_qwen25_coder_32b_ins_0118_merged"

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

if [ ! -d "$MERGED_DIR" ]; then
  mkdir -p "$MERGED_DIR"
fi

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

cd ../ms-swift

# ===================== 训练 =====================
torchrun \
    --nproc_per_node 8 \
    --standalone \
    swift/cli/sft.py \
    --use_hf True \
    \
    --model $MODEL_PATH \
    --train_type lora \
    --lora_rank 128 \
    --lora_alpha 256 \
    --lora_target_modules all-linear \
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
    --learning_rate 1e-4 \
    --lr_scheduler_type "cosine" \
    --report_to none \
    --logging_first_step True \
    --logging_steps 1 \
    \
    --num_train_epochs 1 \
    --gradient_accumulation_steps 16 \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_only_model True \
    --warmup_ratio 0.1 \
    --ddp_backend "nccl" \
    \
    --attn_impl flash_attn

# ===================== 合并导出所有 checkpoint =====================
# 找到 v0-* 目录
V0_DIR=$(ls -dt ${OUTPUT_DIR}/v0-* 2>/dev/null | head -1)

if [ -z "$V0_DIR" ]; then
    echo "未找到 v0-* 目录，退出"
    exit 1
fi

echo "找到训练目录: $V0_DIR"

# 遍历所有 checkpoint 目录
for CKPT in $(ls -d ${V0_DIR}/checkpoint-* 2>/dev/null | sort -V); do
    # 提取 checkpoint 编号，如 checkpoint-500 -> 500
    CKPT_NAME=$(basename $CKPT)
    CKPT_MERGED_DIR="${MERGED_DIR}/${CKPT_NAME}"

    echo "正在合并: $CKPT -> $CKPT_MERGED_DIR"

    swift export \
        --model $MODEL_PATH \
        --adapters $CKPT \
        --output_dir $CKPT_MERGED_DIR \
        --merge_lora true \
        --torch_dtype bfloat16

    echo "完成: $CKPT_MERGED_DIR"
done

echo "所有 checkpoint 合并完成，已导出到: $MERGED_DIR"