#!/usr/bin/env python3
"""
下载 R2E-Gym/R2EGym-SFT-Trajectories 数据集并保存为 JSONL 格式
"""

from datasets import load_dataset
import json


def main():
    print("正在下载数据集 R2E-Gym/R2EGym-SFT-Trajectories ...")

    # 加载数据集
    dataset = load_dataset("R2E-Gym/R2EGym-SFT-Trajectories")

    # 保存为 JSONL 格式
    output_file = "/data/yubo/datasets/R2EGym-Data/R2EGym-SFT-Trajectories.jsonl"

    print(f"正在保存到 {output_file} ...")

    with open(output_file, 'w', encoding='utf-8') as f:
        for split_name in dataset.keys():
            print(f"处理 {split_name} 分片，共 {len(dataset[split_name])} 条记录")
            for item in dataset[split_name]:
                # 添加 split 字段以区分数据来源
                item_with_split = {"split": split_name, **item}
                f.write(json.dumps(item_with_split, ensure_ascii=False) + '\n')

    print(f"完成！数据已保存到 {output_file}")


if __name__ == "__main__":
    main()