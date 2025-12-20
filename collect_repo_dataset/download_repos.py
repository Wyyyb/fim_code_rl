#!/usr/bin/env python3
import csv
import subprocess
import os
from pathlib import Path


def clone_repos(csv_file, output_dir):
    """
    从CSV文件读取git仓库列表并克隆到指定目录

    Args:
        csv_file: CSV文件路径
        output_dir: 输出目录路径
    """
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 读取CSV文件
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            repo_url = row['repo']
            # 从URL中提取仓库名称
            repo_name = repo_url.rstrip('.git').split('/')[-1]
            target_path = output_path / repo_name

            print(f"\n{'=' * 60}")
            print(f"正在处理: {repo_name}")
            print(f"URL: {repo_url}")
            print(f"分类: {row['category']} | 优先级: {row['priority']} | 大小: {row['size']}")
            print(f"目标路径: {target_path}")

            # 检查目录是否已存在
            if target_path.exists():
                print(f"⚠️  目录已存在，跳过克隆")
                continue

            try:
                # 克隆仓库
                result = subprocess.run(
                    ['git', 'clone', repo_url],
                    capture_output=True,
                    text=True,
                    check=True
                )
                print(f"✅ 克隆成功!")

            except subprocess.CalledProcessError as e:
                print(f"❌ 克隆失败!")
                print(f"错误信息: {e.stderr}")
            except FileNotFoundError:
                print("❌ 错误: 未找到git命令，请确保已安装git")
                return

    print(f"\n{'=' * 60}")
    print(f"✅ 所有仓库处理完成!")
    print(f"输出目录: {output_path.absolute()}")


if __name__ == "__main__":
    # 配置参数
    CSV_FILE = "large_repo_groups_113.csv"  # CSV文件路径
    OUTPUT_DIR = "/data/yubo/datasets/collected_sc_1123/"  # 输出目录

    # 执行克隆
    clone_repos(CSV_FILE, OUTPUT_DIR)