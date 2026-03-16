
for i in $(seq 1 50); do
    nohup python step_4_multi_fim_gemini_0227_guided_0316.py \
      -i /data/yubo/datasets/process_data_output_0227/step_3_selected_multi_fim_functions_0227_groups.json \
      -o /data/yubo/datasets/process_data_output_0316/step_4_multi_fim_output_guided_0316.json \
      --shard $i --total-shards 200 --guided \
      > guided_log_0316/guided_shard_$i.log 2>&1 &
    sleep 60
done



