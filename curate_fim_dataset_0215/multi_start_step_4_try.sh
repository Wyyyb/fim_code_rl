

for i in $(seq 31 100); do
    nohup python try_step_4_gemini_fim_0218.py \
      -i /data/yubo/datasets/process_data_output_0215/step_3_selected_fim_functions_0215_functions_sorted_0218.json \
      -o /data/yubo/datasets/process_data_output_0215/try_step_4/step_4_fim_critique_0217_try_0218.json \
      --shard $i --total-shards 200 \
      --wandb --wandb-run-name "exp_0218" \
      > try_shard_$i.log 2>&1 &
    sleep 60
done

