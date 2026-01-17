#!/bin/bash

# 上传所有checkpoint到Hugging Face Hub

REPO_PREFIX="ubowang/fim_qwen25_coder_7b_ins_0116-ckpt"
BASE_DIR="/data/yubo/sft_ckpts/fim_qwen25_coder_7b_ins_0109_ckpt/v2-20260109-221234"

# 按步数排序上传
for ckpt_num in 500 1000 1500 2000 2500 3000 3500 4000 4500 4977; do
    ckpt_dir="$BASE_DIR/checkpoint-$ckpt_num"

    if [ -d "$ckpt_dir" ]; then
        repo_name="${REPO_PREFIX}_${ckpt_num}"

        echo "Uploading $ckpt_dir to $repo_name ..."
        huggingface-cli upload "$repo_name" "$ckpt_dir" --repo-type model

        echo "Done: $repo_name"
        echo "----------------------------------------"
    else
        echo "Warning: $ckpt_dir not found, skipping..."
    fi
done

echo "All checkpoints uploaded!"