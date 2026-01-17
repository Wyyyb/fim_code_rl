#!/bin/bash
set -x

SCRIPT_DIR=$(dirname "$0")
UPLOAD_BASE_DIR="/data2/yubo/sft_ckpts"

for ckpt_step in 500 1000 1500 2000 2500 3000 3500 4000 4500 4977; do
    echo "========================================"
    echo "Starting training for checkpoint-${ckpt_step}"
    echo "========================================"

    bash "$SCRIPT_DIR/single_train.sh" $ckpt_step

    if [ $? -eq 0 ]; then
        echo "Finished training for checkpoint-${ckpt_step}"

        # 上传到HuggingFace
        OUTPUT_DIR="${UPLOAD_BASE_DIR}/fim_qwen25_coder_7b_ins_0116-ckpt_${ckpt_step}_r2egym_sft_0117"
        REPO_NAME="ubowang/fim_qwen25_coder_7b_ins_0116-ckpt_${ckpt_step}_r2egym_sft_0117"

        if [ -d "$OUTPUT_DIR" ]; then
            echo "Uploading $OUTPUT_DIR to $REPO_NAME ..."
            huggingface-cli upload "$REPO_NAME" "$OUTPUT_DIR" --repo-type model
            echo "Upload completed for $REPO_NAME"
        else
            echo "Warning: $OUTPUT_DIR not found, skipping upload..."
        fi
    else
        echo "Error training checkpoint-${ckpt_step}, exiting..."
        exit 1
    fi

    echo ""
done

echo "All training and upload jobs completed!"