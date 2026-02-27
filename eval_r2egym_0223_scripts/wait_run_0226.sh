#!/bin/bash

# 要监控的进程名
PROCESS_NAME="train_fim_7b_0226"
# 进程结束后要执行的脚本
SCRIPT_TO_RUN="sft_r2egym_0226_midtrain_7b_ins.sh"

echo "开始监控进程: $PROCESS_NAME"
echo "检查间隔: 1分钟"
echo "进程结束后将执行: $SCRIPT_TO_RUN"
echo "----------------------------------------"

while true; do
    # 查找进程（排除grep自身和当前监控脚本）
    PROCESS_COUNT=$(ps -ef | grep "$PROCESS_NAME" | grep -v grep | grep -v "$0" | wc -l)

    CURRENT_TIME=$(date '+%Y-%m-%d %H:%M:%S')

    if [ $PROCESS_COUNT -gt 0 ]; then
        echo "[$CURRENT_TIME] 进程 $PROCESS_NAME 正在运行中..."
        # 等待10分钟
        sleep 60
    else
        echo "[$CURRENT_TIME] 进程 $PROCESS_NAME 已结束!"
        echo "[$CURRENT_TIME] 开始执行脚本: $SCRIPT_TO_RUN"

        # 执行目标脚本
        bash "$SCRIPT_TO_RUN"

        echo "[$CURRENT_TIME] 脚本执行完成，监控结束"
        break
    fi
done


cd /data/yubo/sft_ckpts_0215/r2egym_fim_qwen25_coder_7b_ins_0226_mix_489_ckpt

huggingface-cli upload ubowang/r2egym_fim_qwen25_coder_7b_ins_0226_mix_489_ckpt . --repo-type model


