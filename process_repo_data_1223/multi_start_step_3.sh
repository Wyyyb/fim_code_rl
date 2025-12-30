

for i in $(seq 1 50);
do
    nohup python step_3_call_gemini_select_function_1229.py --shard $i --total-shards 50 --wandb --wandb-run-name "exp_1230_50" > step_3_log/shard_$i.log 2>&1 &
done



