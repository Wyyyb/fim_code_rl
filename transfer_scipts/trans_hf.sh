huggingface-cli upload ubowang/ori_qwen25_coder_7b_ins_r2egym_sft_0108-ckpt_808 . --repo-type model
huggingface-cli upload ubowang/fim_qwen25_coder_7b_ins_0105_r2egym_sft_0108-ckpt_808 . --repo-type model

huggingface-cli upload ubowang/ori_qwen25_coder_7b_ins_r2egym_sft_0108-ckpt_400 . --repo-type model
huggingface-cli upload ubowang/fim_qwen25_coder_7b_ins_0105_r2egym_sft_0108-ckpt_400 . --repo-type model


huggingface-cli download ubowang/ori_qwen25_coder_7b_ins_r2egym_sft_0108-ckpt_808 --local-dir /data/yubowang/fim_sft_ckpts/ori_qwen25_coder_7b_ins_r2egym_sft_0108-ckpt_808

huggingface-cli download ubowang/fim_qwen25_coder_7b_ins_0105_r2egym_sft_0108-ckpt_808 --local-dir /data/yubowang/fim_sft_ckpts/fim_qwen25_coder_7b_ins_0105_r2egym_sft_0108-ckpt_808

huggingface-cli download Qwen/Qwen2.5-Coder-32B-Instruct --local-dir /data1/yubo/models/Qwen2.5-Coder-32B-Instruct

huggingface-cli download Qwen/Qwen2.5-Coder-32B-Instruct --local-dir /data2/yubo/models/Qwen2.5-Coder-32B-Instruct

huggingface-cli upload ubowang/fim_midtrain_data_0108_212k /data/yubo/datasets/process_data_output_1228/step_5_sft_data_0105/fim_sft_data_0108.jsonl --repo-type dataset

huggingface-cli download Qwen/Qwen2.5-Coder-14B-Instruct --local-dir /data/yubo/models/Qwen2.5-Coder-14B-Instruct

huggingface-cli download Qwen/Qwen2.5-Coder-14B-Instruct --local-dir /data2/yubo/models/Qwen2.5-Coder-14B-Instruct


huggingface-cli upload ubowang/fim_qwen25_coder_7b_ins_0105_midtrain . --repo-type model

huggingface-cli download Qwen/Qwen3-Coder-30B-A3B-Instruct --local-dir /data/yubo/models/Qwen3-Coder-30B-A3B-Instruct

huggingface-cli download Qwen/Qwen2.5-Coder-14B-Instruct --local-dir /data/yubo/models/Qwen2.5-Coder-14B-Instruct

huggingface-cli upload ubowang/fim_qwen25_coder_7b_ins_0223_midtrain . --repo-type model

huggingface-cli download Qwen/Qwen3.5-35B-A3B --local-dir /data/yubo/models/Qwen3.5-35B-A3B

huggingface-cli upload ubowang/fim_qwen25_coder_7b_ins_midtrain_r2e-gym_pt_0226 . --repo-type model

huggingface-cli upload ubowang/fim_midtrain_data_0226_212k /data/yubo/datasets/process_data_output_0215/step_5_sft_data/fim_midtrain_data_0226.jsonl --repo-type dataset

huggingface-cli upload ubowang/fim_midtrain_data_0226_mix_314k /data/yubo/datasets/process_data_output_0215/step_5_sft_data/fim_sft_data_temp_0226_mix.jsonl --repo-type dataset

