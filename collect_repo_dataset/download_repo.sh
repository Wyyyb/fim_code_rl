#!/bin/bash

# 配置
CSV_FILE="large_repo_groups_113.csv"
OUTPUT_DIR="/data/yubo/datasets/collected_sc_1123"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

echo "开始下载仓库..."
echo "================================"

# 跳过CSV头部，读取每一行
tail -n +2 "$CSV_FILE" | while IFS=',' read -r repo category priority size deps; do
    # 提取仓库名称
    repo_name=$(basename "$repo" .git)
    target_path="$OUTPUT_DIR/$repo_name"

    echo ""
    echo "仓库: $repo_name"
    echo "URL: $repo"
    echo "分类: $category | 优先级: $priority | 大小: $size"

    # 检查目录是否已存在
    if [ -d "$target_path" ]; then
        echo "⚠️  目录已存在，跳过"
        continue
    fi

    # 克隆仓库
    if git clone "$repo" "$target_path"; then
        echo "✅ 克隆成功"
    else
        echo "❌ 克隆失败"
    fi
    echo "--------------------------------"
done

echo ""
echo "================================"
echo "✅ 所有仓库处理完成!"
echo "输出目录: $OUTPUT_DIR"