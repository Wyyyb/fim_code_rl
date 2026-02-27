# 先切分（只需跑一次）
python step_4_multi_fim_gemini_0227.py --pre-shard \
  -i /data/yubo/datasets/process_data_output_0227/step_3_selected_multi_fim_functions_0227_groups.json \
  --total-shards 200

# 再并行跑（每个worker只读自己那个小shard文件）
for i in $(seq 1 50); do
  nohup python step_4_multi_fim_gemini_0227.py \
    -i /data/yubo/datasets/process_data_output_0227/step_3_selected_multi_fim_functions_0227_groups.json \
    -o /data/yubo/datasets/process_data_output_0227/step_4_multi_fim_output_0227.json \
    --shard $i --total-shards 200 > shard_$i.log 2>&1 &
done



