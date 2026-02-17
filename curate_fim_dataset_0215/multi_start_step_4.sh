

for i in $(seq 1 50); do
    nohup python step_4_gemini_fim_and_critique_0217.py \
      -i /data/yubo/datasets/process_data_output_0215/step_3_selected_fim_functions_0215_functions.json \
      -o /data/yubo/datasets/process_data_output_0215/step_4_fim_critique_0217.json \
      --shard $i --total-shards 100 \
      --wandb --wandb-run-name "exp_0217" \
      > shard_$i.log 2>&1 &
    sleep 60
done

