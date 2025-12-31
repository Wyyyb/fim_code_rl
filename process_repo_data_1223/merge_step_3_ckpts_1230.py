#!/usr/bin/env python3
"""
Merge sharded checkpoint files and extract results.
"""

import json
from pathlib import Path


def main():
    input_dir = Path("/data/yubo/datasets/process_data_output_1228/step_3_res_data_1231")
    num_shards = 5

    all_results = []

    for i in range(1, num_shards + 1):
        filename = f"step_3_checkpoint_shard{i}_of_{num_shards}.json"
        path = input_dir / filename

        print(f"📂 Loading {filename}...")

        with open(path, 'r', encoding='utf-8') as f:
            checkpoint = json.load(f)

        results = checkpoint.get('results', [])
        all_results.extend(results)

        # 统计
        num_funcs = sum(len(r.get('selected_function_list', [])) for r in results)
        print(f"   - Results: {len(results)}, Functions: {num_funcs}")

    # 保存
    output_path = input_dir / "step_3_results_merged_1231.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # 总结
    total_funcs = sum(len(r.get('selected_function_list', [])) for r in all_results)
    print(f"\n✅ Saved to {output_path}")
    print(f"   Total results: {len(all_results)}")
    print(f"   Total functions: {total_funcs}")


if __name__ == "__main__":
    main()