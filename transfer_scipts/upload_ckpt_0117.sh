#!/bin/bash

# 上传所有checkpoint到Hugging Face Hub

REPO_PREFIX="ubowang/fim_qwen25_coder_7b_ins_0115_r2egym_sft_0117-ckpt"
BASE_DIR="/data/yubo/sft_ckpts/fim_qwen25_coder_7b_ins_0109_ckpt/v2-20260109-221234"

# 遍历所有checkpoint目录
for ckpt_dir in "$BASE_DIR"/checkpoint-*; do
    if [ -d "$ckpt_dir" ]; then
        # 提取checkpoint步数
        ckpt_num=$(basename "$ckpt_dir" | sed 's/checkpoint-//')
        repo_name="${REPO_PREFIX}_${ckpt_num}"

        echo "Uploading $ckpt_dir to $repo_name ..."
        huggingface-cli upload "$repo_name" "$ckpt_dir" --repo-type model

        echo "Done: $repo_name"
        echo "----------------------------------------"
    fi
done

echo "All checkpoints uploaded!"