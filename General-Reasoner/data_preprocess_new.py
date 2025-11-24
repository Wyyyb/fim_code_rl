import re
import os
import datasets
from datasets import Dataset
import json
from verl.utils.hdfs_io import copy, makedirs
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', default='TIGER-Lab/WebInstruct-verified')
    parser.add_argument('--local-dir', default='~/webinstruct-verified')

    args = parser.parse_args()

    data_source = 'general-reasoner'

    dataset = datasets.load_dataset(args.dataset_name)

    train_dataset = dataset['train']
    test_dataset = dataset['test'].select(range(100))

    instruction_following = "Please reason step by step, and put your final answer within \\boxed{}."


    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop('question')
            question = question_raw + ' ' + instruction_following
            answer = example.pop('answer')
            question_level = example.pop('difficulty')
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "reasoning",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'answer': answer,
                    "question": question_raw,
                    'level': question_level
                }
            }
            return data

        return process_fn


    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    # 保存为 Parquet 格式
    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    # 保存为 JSONL 格式（每行一个 JSON 对象）
    train_dataset.to_json(os.path.join(local_dir, 'train.jsonl'), force_ascii=False)
    test_dataset.to_json(os.path.join(local_dir, 'test.jsonl'), force_ascii=False)

    # 保存为 JSON 格式（使用标准 json 库）
    with open(os.path.join(local_dir, 'train.json'), 'w', encoding='utf-8') as f:
        json.dump(train_dataset.to_list(), f, ensure_ascii=False, indent=2)

    with open(os.path.join(local_dir, 'test.json'), 'w', encoding='utf-8') as f:
        json.dump(test_dataset.to_list(), f, ensure_ascii=False, indent=2)

    print(f"数据已保存到 {local_dir}")
    print(f"- train.parquet, test.parquet (Parquet 格式)")
    print(f"- train.jsonl, test.jsonl (JSONL 格式)")
    print(f"- train.json, test.json (JSON 格式)")