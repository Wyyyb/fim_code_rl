set -x

# conda activate YOUR_ENV

MODEL_PATH="/data/yubo/models/Qwen2.5-Coder-14B-Instruct"

DATA_PATH_1="/data/yubo/datasets/process_data_output_1228/step_5_sft_data_0105/fim_sft_data_0108.jsonl"
DATA_PATH_2="/data/yubo/datasets/process_data_output_0215/step_5_sft_data/fim_sft_data_temp_0219.jsonl"

OUTPUT_DIR="/data/yubo/sft_ckpts_0215/fim_qwen25_coder_14b_ins_0219_ckpt"

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

export CUDA_VISIBLE_DEVICES=4,5,6,7

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
    --dataset $DATA_PATH_1 $DATA_PATH_2 \
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
    --num_train_epochs 1 \
    --gradient_accumulation_steps 32 \
    --save_strategy "steps" \
    --save_steps 1500 \
    --save_only_model True \
    --warmup_ratio 0.1 \
    --ddp_backend "nccl" \
    \
    --freeze_llm False \
    --freeze_vit False \
    --freeze_aligner False\
    --attn_impl flash_attn \

    #     --use_liger_kernel true \
    #     --packing true \
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