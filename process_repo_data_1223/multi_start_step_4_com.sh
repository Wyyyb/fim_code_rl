#!/bin/bash

# 自动发现并启动缺失的 shard

total_shards=50

# 获取正在运行的 shard 列表
running_shards=$(ps -ef | grep step_4_fim_completion_and_critique_1230.py | grep -v grep | grep -oP '\-\-shard \K[0-9]+' | sort -n)

# 找出缺失的 shard
missing_shards=()
for i in $(seq 1 $total_shards); do
  if ! echo "$running_shards" | grep -qw "$i"; then
    missing_shards+=($i)
  fi
done

# 输出信息
echo "正在运行的 shard: $running_shards"
echo "缺失的 shard: ${missing_shards[*]}"
echo "缺失数量: ${#missing_shards[@]}"

# 启动缺失的 shard
if [ ${#missing_shards[@]} -gt 0 ]; then
  for i in "${missing_shards[@]}"; do
    nohup python step_4_fim_completion_and_critique_1230.py \
      -i /data/yubo/datasets/process_data_output_1228/step_3_res_data_1231/step_3_results_merged_1231.json \
      -o /data/yubo/datasets/process_data_output_1228/step_4_res_data_1231/step_4_results_merged_1231.json \
      --shard $i --total-shards 50 \
      --wandb --wandb-run-name "exp_step_4_0101" \
      --skip-preprocess \
      > step_4_0101_logs/shard_$i.log 2>&1 &
    echo "已启动 shard $i"
  done
  echo "共启动 ${#missing_shards[@]} 个缺失的 shard"
else
  echo "所有 shard 都在运行中，无需启动"
fi