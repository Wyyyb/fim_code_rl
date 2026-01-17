huggingface-cli upload ubowang/ori_qwen25_coder_7b_ins_r2egym_sft_0108-ckpt_808 . --repo-type model
huggingface-cli upload ubowang/fim_qwen25_coder_7b_ins_0105_r2egym_sft_0108-ckpt_808 . --repo-type model

huggingface-cli download ubowang/ori_qwen25_coder_7b_ins_r2egym_sft_0108-ckpt_808 --local-dir /data/yubowang/fim_sft_ckpts/ori_qwen25_coder_7b_ins_r2egym_sft_0108-ckpt_808

huggingface-cli download ubowang/fim_qwen25_coder_7b_ins_0105_r2egym_sft_0108-ckpt_808 --local-dir /data/yubowang/fim_sft_ckpts/fim_qwen25_coder_7b_ins_0105_r2egym_sft_0108-ckpt_808


huggingface-cli download Qwen/Qwen2.5-Coder-32B-Instruct --local-dir /data1/yubo/models/Qwen2.5-Coder-32B-Instruct

huggingface-cli download Qwen/Qwen2.5-Coder-32B-Instruct --local-dir /data2/yubo/models/Qwen2.5-Coder-32B-Instruct

huggingface-cli upload ubowang/fim_midtrain_data_0108_212k /data/yubo/datasets/process_data_output_1228/step_5_sft_data_0105/fim_sft_data_0108.jsonl --repo-type dataset

