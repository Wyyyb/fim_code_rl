# Activate the virtual environment (if in new terminal)
cd /data/yubowang/fim_code_rl/R2E-Gym
source .venv/bin/activate

# Set required environment variables
PORT=9002
SERVED_NAME="ori_qwen25_coder_7b_ins_r2egym_sft_0108-ckpt_808"

export TEMP=1
export EXP_NAME="eval_ori_qwen25_coder_7b_ins_r2egym_sft_0108-ckpt_808-0115-0"
export LLM_BASE_URL="http://127.0.0.1:${PORT}/v1"
export OPENAI_API_KEY=EMPTY

# Run the DeepSWE agent on SWE-Bench Verified
time python src/r2egym/agenthub/run/edit.py runagent_multiple \
    --traj_dir "./traj-ori_qwen25_coder_7b_ins_r2egym_sft_0108-ckpt_808_0115-k500" \
    --max_workers 8 \
    --start_idx 0 \
    --k 500 \
    --dataset "R2E-Gym/SWE-Bench-Verified" \
    --split "test" \
    --llm_name "openai/${SERVED_NAME}" \
    --scaffold "r2egym" \
    --use_fn_calling False \
    --exp_name "$EXP_NAME" \
    --temperature "$TEMP" \
    --max_steps_absolute 100 \
    --backend "docker" \
    --condense_history False \
    --max_reward_calc_time 1200 \
    --max_tokens 65536




