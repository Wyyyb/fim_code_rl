
mkdir -p step_4_0101_logs
mkdir -p /data/yubo/datasets/process_data_output_1228/step_4_res_data_1231/

for i in $(seq 1 50); do
  nohup python step_4_fim_completion_and_critique_1230.py \
    -i /data/yubo/datasets/process_data_output_1228/step_3_res_data_1231/step_3_results_merged_1231.json \
    -o /data/yubo/datasets/process_data_output_1228/step_4_res_data_1231/step_4_results_merged_1231.json \
    --shard $i --total-shards 50 \
    --wandb --wandb-run-name "exp_step_4_0101" \
    --skip-preprocess \
    > step_4_0101_logs/shard_$i.log 2>&1 &
done

