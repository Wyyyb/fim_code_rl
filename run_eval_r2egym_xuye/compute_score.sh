
source ../R2E-Gym/.venv/bin/activate

v_res_path="/home/xuye_liu/yubo/fim_code_rl/R2E-Gym/results/ubowang_ori_qwen25_coder_7b_ins_r2egym_sft_0108-ckpt_808_swebv_verified_submission.json"
l_res_path="/home/xuye_liu/yubo/fim_code_rl/R2E-Gym/results/ubowang_ori_qwen25_coder_7b_ins_r2egym_sft_0108-ckpt_808_swebench_lite_submission.json"

cd ../SWE-bench

python -m swebench.harness.run_evaluation \
    --dataset_name princeton-nlp/SWE-bench_Verified \
    --predictions_path "${v_res_path}" \
    --max_workers 32 \
    --run_id swebv \
    --cache_level none


python -m swebench.harness.run_evaluation \
    --dataset_name princeton-nlp/SWE-bench_Lite \
    --predictions_path "${l_res_path}" \
    --max_workers 32 \
    --run_id swebv \
    --cache_level none

