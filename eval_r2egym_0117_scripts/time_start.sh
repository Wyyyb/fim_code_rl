
sleep 16200

cd /data2/yubo/sft_ckpts/ori_qwen25_coder_14b_ins_r2egym_0122/v1-20260123-054036/checkpoint-806/

huggingface-cli upload ubowang/ori_qwen25_coder_14b_ins_r2egym_0122 . --repo-type model

cd /data2/yubo/fim_code_rl/eval_r2egym_0117_scripts

bash sft_r2egym_fim_ckpt_1659_qwen_14b_0122.sh

