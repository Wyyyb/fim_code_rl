
cd /data/yubowang/fim_code_rl/R2E-Gym
source .venv/bin/activate

traj_dir="traj-fim_qwen25_coder_7b_ins_0105_r2egym_sft_0108-ckpt_808-k_10"
traj_file="eval_fim_qwen25_coder_7b_ins_0105_r2egym_sft_0108-ckpt_808-0113-0.jsonl"

uv run python src/r2egym/agenthub/trajectory/create_swebench_submission.py \
    --traj_file_path $traj_dir/$traj_file \
    --output_json_path $traj_dir/swebench_submission.json




