# Activate the virtual environment (if in new terminal)
cd ../R2E-Gym
source .venv/bin/activate

# Set required environment variables
PORT=9002
SERVED_NAME="SWE-agent-LM-7B"

export TEMP=1
export EXP_NAME="eval_SWE-agent-LM-7B"
export LLM_BASE_URL="http://127.0.0.1:${PORT}/v1"
export OPENAI_API_KEY=EMPTY

# Run the DeepSWE agent on SWE-Bench Verified
time python src/r2egym/agenthub/run/edit.py runagent_multiple \
    --traj_dir "./traj_SWE-agent-LM-7B" \
    --max_workers 24 \
    --start_idx 0 \
    --k 500 \
    --dataset "R2E-Gym/SWE-Bench-Verified" \
    --split "test" \
    --llm_name "openai/${SERVED_NAME}" \
    --scaffold "r2egym" \
    --use_fn_calling False \
    --exp_name "$EXP_NAME" \
    --temperature "$TEMP" \
    --max_steps_absolute 500 \
    --backend "docker" \
    --condense_history False \
    --max_reward_calc_time 3600 \
    --max_tokens 131072




