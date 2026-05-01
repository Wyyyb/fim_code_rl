#!/usr/bin/env python3
"""
脚本功能：从源模型仓库下载检查点文件，然后上传到目标模型仓库
源仓库：AgPerry/Qwen2.5-Coder-14B-Instruct-num12_fim-midtrain-sample-10
目标仓库：ubowang/14B-filtered-fim-midtrained-ckpt-0501
"""

import os
from huggingface_hub import snapshot_download, HfApi, login
from pathlib import Path

# 配置参数
SOURCE_REPO = "AgPerry/Qwen2.5-Coder-14B-Instruct-num12_fim-midtrain-sample-10"
TARGET_REPO = "ubowang/14B-filtered-fim-midtrained-ckpt-0501"
LOCAL_DIR = "./temp_model_download"


def main():
    # 步骤 1: 登录 Hugging Face（需要写权限的 token）
    print("请确保你已经设置了 HF_TOKEN 环境变量，或者在下面输入你的 token")
    print("Token 需要有写权限才能上传到目标仓库")

    # 尝试从环境变量获取 token，如果没有则提示用户输入
    token = os.getenv("HF_TOKEN")
    if not token:
        token = input("请输入你的 Hugging Face token (需要写权限): ").strip()

    if token:
        login(token=token)
        print("✓ 已登录 Hugging Face")
    else:
        print("警告：未提供 token，可能无法下载私有模型或上传文件")

    # 步骤 2: 下载源模型
    print(f"\n开始下载模型: {SOURCE_REPO}")
    print(f"保存到本地目录: {LOCAL_DIR}")

    try:
        snapshot_download(
            repo_id=SOURCE_REPO,
            local_dir=LOCAL_DIR,
            local_dir_use_symlinks=False,
            resume_download=True,
            token=token
        )
        print(f"✓ 模型下载完成")
    except Exception as e:
        print(f"✗ 下载失败: {e}")
        return

    # 步骤 3: 上传到目标仓库
    print(f"\n开始上传到目标仓库: {TARGET_REPO}")

    try:
        api = HfApi()

        # 获取所有需要上传的文件
        local_path = Path(LOCAL_DIR)
        files_to_upload = list(local_path.rglob("*"))
        files_to_upload = [f for f in files_to_upload if f.is_file()]

        print(f"找到 {len(files_to_upload)} 个文件需要上传")

        # 逐个上传文件
        for file_path in files_to_upload:
            relative_path = file_path.relative_to(local_path)
            print(f"上传: {relative_path}")

            api.upload_file(
                path_or_fileobj=str(file_path),
                path_in_repo=str(relative_path),
                repo_id=TARGET_REPO,
                repo_type="model",
                token=token
            )

        print(f"✓ 所有文件上传完成到 {TARGET_REPO}")

    except Exception as e:
        print(f"✗ 上传失败: {e}")
        return

    # 步骤 4: 清理临时文件（可选）
    cleanup = input("\n是否删除本地下载的临时文件？(y/n): ").strip().lower()
    if cleanup == 'y':
        import shutil
        shutil.rmtree(LOCAL_DIR)
        print(f"✓ 已删除临时目录: {LOCAL_DIR}")
    else:
        print(f"保留临时文件在: {LOCAL_DIR}")

    print("\n任务完成！")


if __name__ == "__main__":
    # 检查依赖
    try:
        import huggingface_hub
    except ImportError:
        print("错误：需要安装 huggingface_hub 库")
        print("请运行: pip install huggingface_hub")
        exit(1)

    main()
