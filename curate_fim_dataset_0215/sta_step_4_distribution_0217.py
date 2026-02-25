#!/usr/bin/env python
"""
统计所有 shard checkpoint 的各项指标数据分布。

用法示例:
    python analyze_ckpts.py -i "/data/yubo/datasets/process_data_output_0215/step_4_fim_critique_0217_ckpt_shard_*.json"
    python analyze_ckpts.py -i ./ckpts/ --pattern "*.json"
"""

import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def load_ckpt_files(input_path: str, pattern: str = "*.json") -> list[dict]:
    """加载所有 checkpoint 文件"""
    if os.path.isdir(input_path):
        files = sorted(glob.glob(os.path.join(input_path, pattern)))
    else:
        files = sorted(glob.glob(input_path))

    if not files:
        print(f"[ERROR] 未找到任何匹配的文件: {input_path}")
        sys.exit(1)

    print(f"找到 {len(files)} 个 checkpoint 文件")

    all_ckpts = []
    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            all_ckpts.append({"file": f, "data": data})
        except Exception as e:
            print(f"  [WARN] 加载失败: {f} -> {e}")

    print(f"成功加载 {len(all_ckpts)} 个文件\n")
    return all_ckpts


def safe_get(d: dict, *keys, default=None):
    """安全地从嵌套 dict 中取值"""
    for k in keys:
        if isinstance(d, dict):
            d = d.get(k, default)
        else:
            return default
    return d


def compute_numeric_stats(values: list, name: str) -> dict:
    """计算数值型指标的统计信息"""
    arr = np.array([v for v in values if v is not None])
    if len(arr) == 0:
        return {"name": name, "count": 0}
    return {
        "name": name,
        "count": len(arr),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def print_numeric_stats(stats: dict):
    """打印数值型统计结果"""
    if stats["count"] == 0:
        print(f"  {stats['name']}: (无数据)")
        return
    print(
        f"  {stats['name']:.<40s} "
        f"n={stats['count']:<7d} "
        f"mean={stats['mean']:<9.3f} "
        f"std={stats['std']:<9.3f} "
        f"min={stats['min']:<9.3f} "
        f"p25={stats['p25']:<9.3f} "
        f"med={stats['median']:<9.3f} "
        f"p75={stats['p75']:<9.3f} "
        f"p90={stats['p90']:<9.3f} "
        f"p95={stats['p95']:<9.3f} "
        f"max={stats['max']:<9.3f}"
    )


def print_counter(counter: Counter, name: str, total: int):
    """打印分类型统计结果"""
    print(f"  {name} (共 {total} 条):")
    for key, cnt in counter.most_common():
        pct = cnt / total * 100 if total > 0 else 0
        print(f"    {str(key):.<50s} {cnt:>6d}  ({pct:5.1f}%)")


def analyze(all_ckpts: list[dict]):
    """主分析逻辑"""
    # ========== 1. Shard 级别统计 ==========
    print("=" * 100)
    print("1. Shard 级别概览")
    print("=" * 100)

    shard_result_counts = []
    total_results = 0
    total_processed_ids = 0

    for ckpt in all_ckpts:
        data = ckpt["data"]
        n_processed = len(data.get("processed_ids", []))
        n_results = len(data.get("results", []))
        total_processed_ids += n_processed
        total_results += n_results
        shard_result_counts.append(n_results)
        basename = os.path.basename(ckpt["file"])
        # 不逐个打印，只统计

    print(f"  总 processed_ids 数: {total_processed_ids}")
    print(f"  总 results 数:       {total_results}")
    print_numeric_stats(compute_numeric_stats(shard_result_counts, "每个 shard 的 results 数"))
    print()

    # ========== 收集所有 results ==========
    all_results = []
    for ckpt in all_ckpts:
        all_results.extend(ckpt["data"].get("results", []))

    if not all_results:
        print("[WARN] 没有任何 results 数据")
        return

    # ========== 2. 函数基本属性 ==========
    print("=" * 100)
    print("2. 函数基本属性分布")
    print("=" * 100)

    numeric_fields = [
        ("loc", "LOC (行数)"),
        ("complexity", "Complexity (复杂度)"),
        ("inferability", "Inferability (可推断性)"),
        ("fim_score", "FIM Score"),
        ("difficulty", "Difficulty (难度)"),
        ("file_lines", "File Lines (文件总行数)"),
    ]
    for field, label in numeric_fields:
        values = [r.get(field) for r in all_results]
        print_numeric_stats(compute_numeric_stats(values, label))

    # graph_stats
    graph_fields = [
        ("graph_stats.total_functions", "Graph: total_functions"),
        ("graph_stats.call_edges", "Graph: call_edges"),
        ("graph_stats.sibling_pairs", "Graph: sibling_pairs"),
    ]
    for field, label in graph_fields:
        keys = field.split(".")
        values = [safe_get(r, *keys) for r in all_results]
        print_numeric_stats(compute_numeric_stats(values, label))
    print()

    # ========== 3. 分类属性分布 ==========
    print("=" * 100)
    print("3. 分类属性分布")
    print("=" * 100)

    categorical_fields = [
        ("category", "Category"),
        ("quality_rating", "Quality Rating"),
        ("stars_estimate", "Stars Estimate"),
    ]
    for field, label in categorical_fields:
        counter = Counter(r.get(field, "N/A") for r in all_results)
        print_counter(counter, label, len(all_results))
        print()

    # ========== 4. Feasibility & Discard ==========
    print("=" * 100)
    print("4. Feasibility & Discard 统计")
    print("=" * 100)

    feasible_counter = Counter()
    discard_counter = Counter()
    feasibility_confidence = []

    for r in all_results:
        feasible_counter[r.get("is_feasible", "N/A")] += 1
        discard_counter[r.get("should_discard", "N/A")] += 1
        conf = safe_get(r, "critique_response", "feasibility", "confidence")
        if conf is not None:
            feasibility_confidence.append(conf)

    print_counter(feasible_counter, "is_feasible", len(all_results))
    print()
    print_counter(discard_counter, "should_discard", len(all_results))
    print()
    print_numeric_stats(compute_numeric_stats(feasibility_confidence, "Feasibility Confidence"))
    print()

    # ========== 5. FIM Response 统计 ==========
    print("=" * 100)
    print("5. FIM Response 统计")
    print("=" * 100)

    fim_parse_counter = Counter()
    fim_input_tokens = []
    fim_output_tokens = []
    fim_latency = []

    for r in all_results:
        fim = r.get("fim_response", {})
        if not fim:
            continue
        fim_parse_counter[fim.get("parse_success", "N/A")] += 1

        tu = fim.get("token_usage", {})
        if tu.get("input_tokens") is not None:
            fim_input_tokens.append(tu["input_tokens"])
        if tu.get("output_tokens") is not None:
            fim_output_tokens.append(tu["output_tokens"])
        if tu.get("latency") is not None:
            fim_latency.append(tu["latency"])

    print_counter(fim_parse_counter, "FIM parse_success", len(all_results))
    print()
    print_numeric_stats(compute_numeric_stats(fim_input_tokens, "FIM input_tokens"))
    print_numeric_stats(compute_numeric_stats(fim_output_tokens, "FIM output_tokens"))
    print_numeric_stats(compute_numeric_stats(fim_latency, "FIM latency (s)"))
    print()

    # ========== 6. Critique Response 统计 ==========
    print("=" * 100)
    print("6. Critique Response 统计")
    print("=" * 100)

    critique_parse_counter = Counter()
    critique_input_tokens = []
    critique_output_tokens = []
    critique_latency = []

    score_fields = [
        "correctness",
        "executability",
        "api_usage",
        "readability",
        "completeness",
    ]
    score_values = defaultdict(list)
    overall_scores = []

    for r in all_results:
        crit = r.get("critique_response", {})
        if not crit:
            continue
        critique_parse_counter[crit.get("parse_success", "N/A")] += 1

        tu = crit.get("token_usage", {})
        if tu.get("input_tokens") is not None:
            critique_input_tokens.append(tu["input_tokens"])
        if tu.get("output_tokens") is not None:
            critique_output_tokens.append(tu["output_tokens"])
        if tu.get("latency") is not None:
            critique_latency.append(tu["latency"])

        scores = crit.get("scores", {})
        for sf in score_fields:
            v = scores.get(sf)
            if v is not None:
                score_values[sf].append(v)

        os_val = crit.get("overall_score")
        if os_val is not None:
            overall_scores.append(os_val)

    print_counter(critique_parse_counter, "Critique parse_success", len(all_results))
    print()
    print_numeric_stats(compute_numeric_stats(critique_input_tokens, "Critique input_tokens"))
    print_numeric_stats(compute_numeric_stats(critique_output_tokens, "Critique output_tokens"))
    print_numeric_stats(compute_numeric_stats(critique_latency, "Critique latency (s)"))
    print()

    print("  --- Critique 评分分布 ---")
    for sf in score_fields:
        print_numeric_stats(compute_numeric_stats(score_values[sf], f"scores.{sf}"))
    print_numeric_stats(compute_numeric_stats(overall_scores, "overall_score"))
    print()

    # 评分的离散分布
    print("  --- Critique 评分离散分布 ---")
    for sf in score_fields:
        counter = Counter(int(v) for v in score_values[sf])
        print_counter(counter, f"scores.{sf}", len(score_values[sf]))
        print()
    counter = Counter(int(v) for v in overall_scores)
    print_counter(counter, "overall_score", len(overall_scores))
    print()

    # ========== 7. Token Usage 汇总 ==========
    print("=" * 100)
    print("7. Token Usage 全局汇总")
    print("=" * 100)

    total_fim_in = 0
    total_fim_out = 0
    total_crit_in = 0
    total_crit_out = 0
    total_requests = 0

    for ckpt in all_ckpts:
        tu = ckpt["data"].get("token_usage", {})
        total_fim_in += tu.get("fim_input_tokens", 0)
        total_fim_out += tu.get("fim_output_tokens", 0)
        total_crit_in += tu.get("critique_input_tokens", 0)
        total_crit_out += tu.get("critique_output_tokens", 0)
        total_requests += tu.get("request_count", 0)

    total_in = total_fim_in + total_crit_in
    total_out = total_fim_out + total_crit_out

    print(f"  总请求数:             {total_requests:>12,d}")
    print(f"  FIM input tokens:     {total_fim_in:>12,d}")
    print(f"  FIM output tokens:    {total_fim_out:>12,d}")
    print(f"  Critique input tokens:{total_crit_in:>12,d}")
    print(f"  Critique output tokens:{total_crit_out:>12,d}")
    print(f"  总 input tokens:      {total_in:>12,d}")
    print(f"  总 output tokens:     {total_out:>12,d}")
    print(f"  总 tokens:            {total_in + total_out:>12,d}")
    print()

    # ========== 8. Discard Reason 统计 ==========
    # print("=" * 100)
    # print("8. Discard Reason 统计")
    # print("=" * 100)
    #
    # discard_reasons = Counter()
    # for r in all_results:
    #     reason = r.get("discard_reason", "")
    #     if r.get("should_discard", False):
    #         discard_reasons[reason if reason else "(empty)"] += 1
    #
    # n_discarded = sum(discard_reasons.values())
    # if n_discarded > 0:
    #     print_counter(discard_reasons, "Discard Reasons (仅 should_discard=True)", n_discarded)
    # else:
    #     print("  无需丢弃的样本")
    print()


def main():
    parser = argparse.ArgumentParser(description="统计所有 shard checkpoint 的各项指标")
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="checkpoint 文件路径 (支持 glob pattern) 或目录路径",
    )
    parser.add_argument(
        "--pattern",
        default="*.json",
        help="当 --input 为目录时，文件匹配模式 (默认: *.json)",
    )
    args = parser.parse_args()

    all_ckpts = load_ckpt_files(args.input, args.pattern)
    analyze(all_ckpts)


if __name__ == "__main__":

    # python sta_step_4_distribution_0217.py -i "/data/yubo/datasets/process_data_output_0215/step_4_fim_critique_0217_checkpoint_shard*200.json"
    # python sta_step_4_distribution_0217.py -i "/data/yubo/datasets/process_data_output_0215/try_step_4/step_4_fim_critique_0217_try_0218_checkpoint_shard*.json"
    main()