from datasets import load_dataset
from transformers import AutoTokenizer
from collections import Counter
import numpy as np

# 加载数据集
dataset = load_dataset("R2E-Gym/R2EGym-SFT-Trajectories", split="train")

# 加载 tokenizer（用 Qwen-2.5-Coder-7B 的 tokenizer）
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")

def count_tokens(messages):
    """计算一条对话的总 token 数"""
    total_tokens = 0
    for msg in messages:
        content = msg.get("content", "")
        total_tokens += len(tokenizer.encode(content, add_special_tokens=False))
    return total_tokens

# 统计所有样本的 token 数
token_counts = []
for i, example in enumerate(dataset):
    tokens = count_tokens(example["messages"])
    token_counts.append(tokens)
    if (i + 1) % 100 == 0:
        print(f"已处理 {i + 1}/{len(dataset)} 条样本")

# 统计结果
token_counts = np.array(token_counts)

print("\n" + "=" * 50)
print("Token 数量分布统计")
print("=" * 50)
print(f"样本总数: {len(token_counts)}")
print(f"最小值: {token_counts.min()}")
print(f"最大值: {token_counts.max()}")
print(f"平均值: {token_counts.mean():.2f}")
print(f"中位数: {np.median(token_counts):.2f}")
print(f"标准差: {token_counts.std():.2f}")

# 分位数
print("\n分位数:")
for p in [25, 50, 75, 90, 95, 99]:
    print(f"  {p}%: {np.percentile(token_counts, p):.0f}")

# 按区间统计
print("\n按区间分布:")
bins = [0, 4000, 8000, 16000, 32000, 64000, 128000, float('inf')]
labels = ["0-4K", "4K-8K", "8K-16K", "16K-32K", "32K-64K", "64K-128K", "128K+"]
for i in range(len(bins) - 1):
    count = np.sum((token_counts >= bins[i]) & (token_counts < bins[i + 1]))
    pct = count / len(token_counts) * 100
    print(f"  {labels[i]}: {count} ({pct:.2f}%)")

# 可选：保存详细结果到文件
np.save("token_counts.npy", token_counts)
print("\n详细 token 数已保存到 token_counts.npy")