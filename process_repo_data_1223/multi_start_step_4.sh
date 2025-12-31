
mkdir -p step_4_test_logs

for i in $(seq 1 5); do
  nohup python step_4_fim_completion_and_critique_1230.py \
    -i /data/yubo/datasets/process_data_output_1228/step_3_bk_1229/step_3_results_merged_1230.json \
    -o /data/yubo/datasets/process_data_output_1228/step_3_bk_1229/step_4_results_merged_1230_test.json \
    --shard $i --total-shards 5 \
    --wandb --wandb-run-name "exp_step_4_1230_test" \
    > step_4_test_logs/shard_$i.log 2>&1 &
done