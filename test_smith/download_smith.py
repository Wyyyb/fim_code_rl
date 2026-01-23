from datasets import load_dataset
import json
import tiktoken

# 加载数据集的 xml split
print("正在加载数据集...")
dataset = load_dataset("SWE-bench/SWE-smith-trajectories", split="xml")

# 初始化 tokenizer (使用 cl100k_base，适用于 GPT-4/Claude 等模型)
enc = tiktoken.get_encoding("cl100k_base")

# 保存路径
output_file = "../local_data/swe_smith_xml_messages.jsonl"

token_counts = []

print(f"正在处理 {len(dataset)} 条数据...")

with open(output_file, "w", encoding="utf-8") as f:
    for i, item in enumerate(dataset):
        messages = item["messages"]
        resolved = item["resolved"]
        if resolved is False:
            continue
        # 计算 token 数（将 messages 序列化后计算）
        messages_str = json.dumps(messages, ensure_ascii=False)
        num_tokens = len(enc.encode(messages_str))
        token_counts.append(num_tokens)

        # 写入 jsonl
        f.write(json.dumps(messages, ensure_ascii=False) + "\n")

        if (i + 1) % 1000 == 0:
            print(f"已处理 {i + 1} 条")


# 统计信息
print("\n" + "=" * 50)
print("统计结果：")
print(f"总条数: {len(token_counts)}")
print(f"总 token 数: {sum(token_counts):,}")
print(f"平均 token 数: {sum(token_counts) / len(token_counts):,.1f}")
print(f"最小 token 数: {min(token_counts):,}")
print(f"最大 token 数: {max(token_counts):,}")
print(f"\n已保存到: {output_file}")

# 保存 token 统计
stats_file = "swe_smith_xml_token_stats.json"
with open(stats_file, "w") as f:
    json.dump({
        "total_samples": len(token_counts),
        "total_tokens": sum(token_counts),
        "avg_tokens": sum(token_counts) / len(token_counts),
        "min_tokens": min(token_counts),
        "max_tokens": max(token_counts),
        "token_counts": token_counts
    }, f, indent=2)
print(f"Token 统计已保存到: {stats_file}")