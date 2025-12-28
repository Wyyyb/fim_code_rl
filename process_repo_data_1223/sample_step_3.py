import json
import random

# 读取原始数据
input_path = "/data/yubo/datasets/fim_training_data.json"
output_path = "/data/yubo/datasets/fim_training_data_sample_5.json"

with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 随机采样5条数据
sample_size = 5
if len(data) >= sample_size:
    sampled_data = random.sample(data, sample_size)
else:
    sampled_data = data
    print(f"警告：原始数据只有 {len(data)} 条，少于请求的 {sample_size} 条")

# 保存采样数据
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(sampled_data, f, ensure_ascii=False, indent=2)

print(f"已从 {len(data)} 条数据中采样 {len(sampled_data)} 条")
print(f"采样结果已保存至: {output_path}")