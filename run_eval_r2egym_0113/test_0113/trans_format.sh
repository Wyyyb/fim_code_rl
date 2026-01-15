
cd /data/yubowang/fim_code_rl/R2E-Gym
source .venv/bin/activate

traj_dir="traj-fim_qwen25_coder_7b_ins_0105_r2egym_sft_0108-ckpt_808_0115-k500"
traj_file="eval_fim_qwen25_coder_7b_ins_0105_r2egym_sft_0108-ckpt_808-0115-0.jsonl"

python src/r2egym/agenthub/trajectory/create_swebench_submission.py \
    --traj_file_path $traj_dir/$traj_file \
    --output_json_path $traj_dir/swebench_submission.json


cd ../SWE-bench

python -m swebench.harness.run_evaluation \
    --dataset_name princeton-nlp/SWE-bench_Verified \
    --predictions_path ../R2E-Gym/$traj_dir/swebench_submission.json \
    --max_workers 32 \
    --run_id swebv \
    --cache_level none


