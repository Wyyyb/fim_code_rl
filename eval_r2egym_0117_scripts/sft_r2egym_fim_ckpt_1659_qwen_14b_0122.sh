set -x

# conda activate YOUR_ENV

MODEL_PATH="/data2/yubo/sft_ckpts/fim_qwen25_coder_14b_ins_0118_ckpt/v1-20260119-080524/checkpoint-1659"

DATA_PATH="/data2/yubo/datasets/R2EGym-Data/R2EGym-SFT-Trajectories.jsonl"

OUTPUT_DIR="/data2/yubo/sft_ckpts/fim_ckpt_1659_qwen25_coder_14b_ins_r2egym_0122"

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

cd ../ms-swift

torchrun \
    --nproc_per_node 8 \
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
    --gradient_accumulation_steps 1 \
    --save_strategy "steps" \
    --save_steps 400 \
    --save_only_model True \
    --warmup_ratio 0.05 \
    --ddp_backend "nccl" \
    \
    --freeze_llm False \
    --freeze_vit False \
    --freeze_aligner False\
    --attn_impl flash_attn \

    # --attn_impl flash_attn \

    # --save_strategy "epoch" \
    # --save_strategy "steps" \
    # --save_steps 109 \
    # --save_total_limit 5 \
    # --deepspeed zero3 \
    # --max_steps -1 \
    # --device_map auto \
    # --override_exist_file True \
    # --eval_strategy None \
    # --custom_dataset_in