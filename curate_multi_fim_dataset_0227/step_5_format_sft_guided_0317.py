#!/usr/bin/env python3
"""
Multi-Function SFT Training Data Preparation Script

Processes shard checkpoint files from the multi-function FIM pipeline
(step_4_multi_fim_gemini.py) and generates SFT training data.

Key differences from single-function version:
  - Reads from *_pairs.json and *_triples.json shard files separately
  - Filters on BOTH per-function scores AND group coherence score
  - If ANY function in a group fails quality thresholds, the entire group
    is discarded (one bad function poisons the group)
  - Prompt template matches the multi-function FIM completion prompt
  - Outputs two separate JSONL files for pairs and triples
  - Statistics include group-type breakdown and coherence score distribution

Input file naming convention:
  step_4_multi_fim_output_0227_shard{N}_of_{M}_pairs.json
  step_4_multi_fim_output_0227_shard{N}_of_{M}_triples.json

  where the checkpoint structure contains 'results_pair' / 'results_triple'
  keys, or is a flat list of group records.

Usage:
  python prepare_multi_sft_data.py \\
    -d /data/yubo/datasets/process_data_output_0227/ \\
    --pairs-pattern "step_4_multi_fim_output_0227_shard*_of_*_pairs.json" \\
    --triples-pattern "step_4_multi_fim_output_0227_shard*_of_*_triples.json" \\
    --output-pairs /data/yubo/datasets/multi_fim_sft_pairs.jsonl \\
    --output-triples /data/yubo/datasets/multi_fim_sft_triples.jsonl
"""

import os
import json
import math
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Optional, List, Dict, Any, Tuple

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    print("⚠️ tiktoken not installed. Token counting will use approximation.")


# =============================================================================
# Prompt Template (must match step_4_multi_fim_gemini.py)
# =============================================================================

GROUP_TYPE_DESCRIPTIONS = {
    "caller_callee": "have a caller-callee relationship (one calls the other)",
    "co_callee": "are both called by the same function (co-callees)",
    "sibling_coupled": "are methods of the same class that share instance variables",
    "mutual_call": "call each other (mutual dependency)",
    "call_chain": "form a call chain (A calls B, B calls C)",
    "hub": "have a hub pattern (one function calls the other two)",
    "fan_in": "have a fan-in pattern (two functions both call the third)",
    "class_triad": "are three methods of the same class sharing instance variables",
}

MULTI_FIM_COMPLETION_PROMPT = '''You are an expert Python programmer. Your task is to complete {num_functions} masked functions based on the surrounding code context.

## Task

Below is a Python file where {num_functions} function bodies have been replaced with `# <MASKED_FUNCTION_BODY>`. These functions are **structurally related** — they {relationship_description}.

Your job is to:

1. **Analyze the context**: Look at the imports, other functions, class definitions, and how these functions interact with the rest of the code and with EACH OTHER.
2. **Reason about cross-function consistency**: These functions are related. Think about:
   - How they call each other or are called together
   - Whether they share instance variables or state
   - Whether they need to maintain consistent interfaces (parameter passing, return types)
   - The overall design pattern they implement together
3. **Write the complete function bodies**: Provide working implementations for ALL {num_functions} functions.

## Output Format

Please structure your response as follows:

### Reasoning
<Your step-by-step analysis of what each function should do, and how they relate to each other>

{function_output_sections}

## Important Notes
- Provide the function BODY only (the code inside the function), not the signature
- Maintain proper indentation (the body should be indented with 4 spaces or appropriate level)
- The implementations should be **mutually consistent** — if function A calls function B, make sure the call matches B's actual implementation
- If a function has a docstring, include it as part of the body
- Pay special attention to shared state (e.g., self.xxx attributes) being used consistently across functions

## Code with Masked Functions

```python
{masked_code}
```

## Functions to Complete

{function_list}

Please analyze the context and provide your reasoning followed by the implementations.
'''


# Special tokens to remove from training data
SPECIAL_TOKENS_TO_REMOVE = [
    '<|endoftext|>', '<|fim_prefix|>', '<|fim_middle|>', '<|fim_suffix|>',
    '<|fim▁begin|>', '<|fim▁hole|>', '<|fim▁end|>',
    '<|pad|>', '<|eos|>', '<|bos|>', '<|sep|>', '<|cls|>', '<|mask|>',
]


def clean_special_tokens(text: str) -> str:
    """Remove special tokens from text that could interfere with training."""
    for token in SPECIAL_TOKENS_TO_REMOVE:
        text = text.replace(token, '')
    return text


# =============================================================================
# Loading
# =============================================================================

def load_shard_files(
    checkpoint_dir: str,
    pattern: str,
    results_key: Optional[str] = None,
) -> List[Dict]:
    """
    Load all shard files matching a glob pattern.

    Handles two formats:
      1. Checkpoint format: JSON with a top-level key (e.g. 'results_pair')
         containing the list of records.
      2. Flat list format: JSON is directly a list of records.

    Args:
        checkpoint_dir: Directory containing shard files.
        pattern: Glob pattern for shard files.
        results_key: If set, look for this key in the JSON (checkpoint format).
                     If None, try common keys then fall back to flat list.

    Returns:
        Merged list of all records across all shard files.
    """
    ckpt_path = Path(checkpoint_dir)
    all_records: List[Dict] = []
    loaded_files = 0

    shard_files = sorted(ckpt_path.glob(pattern))
    if not shard_files:
        print(f"⚠️  No files matching '{pattern}' in {checkpoint_dir}")
        return all_records

    print(f"📂 Found {len(shard_files)} shard files for pattern: {pattern}")

    for sf in shard_files:
        try:
            with open(sf, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Determine where the records are
            records = None
            if isinstance(data, list):
                records = data
            elif isinstance(data, dict):
                # Try explicit key first
                if results_key and results_key in data:
                    records = data[results_key]
                else:
                    # Try common keys from checkpoint format
                    for key in ['results_pair', 'results_triple', 'results']:
                        if key in data and isinstance(data[key], list):
                            records = data[key]
                            break
                if records is None:
                    print(f"   ⚠️  {sf.name}: no recognized records key, skipping")
                    continue

            all_records.extend(records)
            loaded_files += 1
            print(f"   ✅ {len(records):>5d} records from {sf.name}")

        except Exception as e:
            print(f"   ❌ Failed to load {sf.name}: {e}")

    print(f"   📊 Total: {len(all_records):,} records from {loaded_files} files\n")
    return all_records


# =============================================================================
# Filtering
# =============================================================================

def filter_multi_fim_samples(
    records: List[Dict],
    min_per_func_score: int = 3,
    min_group_overall_score: int = 4,
    min_coherence_score: int = 3,
) -> Tuple[List[Dict], Dict[str, int]]:
    """
    Filter out low-quality multi-function group samples.

    A group is DISCARDED if ANY of:
      - critique parse failed
      - FIM parse failed (no implementations extracted)
      - should_discard == True (set by Gemini critique)
      - any per-function overall_score < min_per_func_score
      - any per-function executability <= 1
      - group_overall_score < min_group_overall_score
      - group coherence_score < min_coherence_score

    Args:
        records: List of group records from shard files.
        min_per_func_score: Minimum per-function overall_score (inclusive).
        min_group_overall_score: Minimum group_overall_score (inclusive).
        min_coherence_score: Minimum group coherence_score (inclusive).

    Returns:
        (filtered_records, filter_stats_dict)
    """
    stats = {
        'total_input': len(records),
        'passed': 0,
        'filtered_no_critique': 0,
        'filtered_critique_parse_failed': 0,
        'filtered_no_fim': 0,
        'filtered_fim_parse_failed': 0,
        'filtered_should_discard': 0,
        'filtered_low_per_func_score': 0,
        'filtered_low_per_func_executability': 0,
        'filtered_low_group_overall': 0,
        'filtered_low_coherence': 0,
    }

    filtered = []

    for record in records:
        # --- FIM response checks ---
        fim = record.get('fim_response', {})
        if not fim:
            stats['filtered_no_fim'] += 1
            continue
        if not fim.get('parse_success', False):
            stats['filtered_fim_parse_failed'] += 1
            continue
        implementations = fim.get('implementations', {})
        functions = record.get('functions', [])
        if not implementations or len(implementations) == 0:
            stats['filtered_fim_parse_failed'] += 1
            continue

        # --- Critique response checks ---
        critique = record.get('critique_response', {})
        if not critique:
            stats['filtered_no_critique'] += 1
            continue
        if not critique.get('parse_success', False):
            stats['filtered_critique_parse_failed'] += 1
            continue

        # --- Gemini's own discard decision ---
        if record.get('should_discard', False) is True:
            stats['filtered_should_discard'] += 1
            continue

        # --- Per-function quality checks ---
        per_func = critique.get('per_function', [])
        per_func_failed = False
        for pf in per_func:
            scores = pf.get('scores', {})
            # Executability hard floor
            if scores.get('executability', 0) <= 1:
                stats['filtered_low_per_func_executability'] += 1
                per_func_failed = True
                break
            # Per-function overall score
            if pf.get('overall_score', 0) < min_per_func_score:
                stats['filtered_low_per_func_score'] += 1
                per_func_failed = True
                break
        if per_func_failed:
            continue

        # --- Group overall score ---
        group_overall = critique.get('group_overall_score', 0)
        if isinstance(group_overall, (int, float)) and group_overall < min_group_overall_score:
            stats['filtered_low_group_overall'] += 1
            continue

        # --- Group coherence score ---
        coherence = critique.get('group_coherence', {})
        coherence_score = coherence.get('coherence_score', 0)
        if isinstance(coherence_score, (int, float)) and coherence_score < min_coherence_score:
            stats['filtered_low_coherence'] += 1
            continue

        # Passed all filters
        filtered.append(record)
        stats['passed'] += 1

    return filtered, stats


# =============================================================================
# SFT Sample Construction
# =============================================================================

def _build_function_output_sections(functions: List[Dict]) -> str:
    """Build per-function output format sections for the prompt."""
    sections = []
    for i, fn in enumerate(functions, 1):
        name = fn.get('func_name', f'function_{i}')
        sections.append(
            f"### Function {i}: `{name}`\n"
            f"```python\n<The complete function body for {name}>\n```"
        )
    return "\n\n".join(sections)


def _build_function_list(functions: List[Dict]) -> str:
    """Build the numbered function list for the prompt."""
    lines = []
    for i, fn in enumerate(functions, 1):
        name = fn.get('func_name', '')
        loc = fn.get('loc', '?')
        lines.append(f"{i}. `{name}` (approx. {loc} lines)")
    return "\n".join(lines)


def construct_multi_sft_sample(record: Dict) -> Optional[Dict]:
    """
    Construct a single SFT training sample from a multi-function group record.

    The user message is the multi-FIM prompt (with masked code + function list).
    The assistant message is Gemini's raw response (reasoning + implementations).

    Returns:
        Dict with 'messages' key in chat format, plus metadata fields,
        or None if construction fails.
    """
    try:
        masked_code = record.get('masked_code_content', '')
        functions = record.get('functions', [])
        group_type = record.get('group_type', '')
        group_size = record.get('group_size', len(functions))
        fim = record.get('fim_response', {})

        if not masked_code or not functions:
            return None

        relationship_desc = GROUP_TYPE_DESCRIPTIONS.get(
            group_type, "are structurally related"
        )

        # Build user message (same prompt as step 4)
        user_content = MULTI_FIM_COMPLETION_PROMPT.format(
            num_functions=group_size,
            relationship_description=relationship_desc,
            function_output_sections=_build_function_output_sections(functions),
            masked_code=masked_code,
            function_list=_build_function_list(functions),
        )

        # Assistant message = Gemini's raw response (contains reasoning + code)
        assistant_content = fim.get('raw_response', '')
        if not assistant_content:
            return None

        # Clean special tokens
        user_content = clean_special_tokens(user_content)
        assistant_content = clean_special_tokens(assistant_content)

        # Build the SFT sample
        sample = {
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ],
            # Metadata (not used by trainer, but useful for analysis/filtering)
            "metadata": {
                # --- Repo-level fields ---
                "repo_id": record.get('repo_id', ''),
                "repository_url": record.get('repository_url', ''),
                "file_path": record.get('file_path', ''),
                "line_num": record.get('line_num', ''),
                "func_num": record.get('func_num', ''),
                "category": record.get('category', ''),
                "quality_rating": record.get('quality_rating', ''),
                "description": record.get('description', ''),
                "stars_estimate": record.get('stars_estimate', ''),
                "notes": record.get('notes', ''),
                # --- Group-level fields ---
                "sample_id": record.get('sample_id', ''),
                "group_type": group_type,
                "group_size": group_size,
                "coupling": record.get('coupling', 0),
                "group_score": record.get('group_score', 0),
                "group_difficulty": record.get('group_difficulty', 0),
                "group_overall_score": record.get('group_overall_score', 0),
                "group_coherence_score": record.get(
                    'critique_response', {}
                ).get('group_coherence', {}).get('coherence_score', 0),
                "func_names": [f.get('func_name', '') for f in functions],
            },
        }
        return sample

    except Exception as e:
        uid = record.get('unique_id', 'unknown')
        print(f"   ⚠️  Failed to construct sample for {uid}: {e}")
        return None


# =============================================================================
# Token Counting
# =============================================================================

_ENCODING = None

def count_tokens(text: str) -> int:
    """Count tokens using tiktoken (cl100k_base) or fallback to char/4."""
    global _ENCODING
    if TIKTOKEN_AVAILABLE:
        if _ENCODING is None:
            _ENCODING = tiktoken.get_encoding("cl100k_base")
        return len(_ENCODING.encode(text, disallowed_special=()))
    return len(text) // 4


# =============================================================================
# Statistics
# =============================================================================

def compute_statistics(
    filtered_records: List[Dict],
    sft_samples: List[Dict],
    label: str = "",
) -> Dict[str, Any]:
    """Compute detailed statistics for a set of filtered records and SFT samples."""
    stats: Dict[str, Any] = {
        'label': label,
        'total_samples': len(sft_samples),
        'group_type_counts': defaultdict(int),
        'per_func_score_distributions': {
            'correctness': defaultdict(int),
            'executability': defaultdict(int),
            'api_usage': defaultdict(int),
            'completeness': defaultdict(int),
            'per_func_overall': defaultdict(int),
        },
        'group_overall_distribution': defaultdict(int),
        'coherence_distribution': defaultdict(int),
        'token_stats': {},
    }

    # Group type breakdown
    for rec in filtered_records:
        gt = rec.get('group_type', '?')
        stats['group_type_counts'][gt] += 1

    # Score distributions
    for rec in filtered_records:
        critique = rec.get('critique_response', {})
        # Group-level scores
        g_overall = critique.get('group_overall_score', 0)
        stats['group_overall_distribution'][g_overall] += 1

        coh = critique.get('group_coherence', {}).get('coherence_score', 0)
        # Round to int for bucketing
        stats['coherence_distribution'][int(round(coh))] += 1

        # Per-function scores
        for pf in critique.get('per_function', []):
            scores = pf.get('scores', {})
            for key in ['correctness', 'executability', 'api_usage', 'completeness']:
                val = scores.get(key, 0)
                stats['per_func_score_distributions'][key][val] += 1
            pf_overall = pf.get('overall_score', 0)
            stats['per_func_score_distributions']['per_func_overall'][pf_overall] += 1

    # Token statistics
    input_tokens_list = []
    output_tokens_list = []
    for sample in sft_samples:
        msgs = sample.get('messages', [])
        if len(msgs) >= 2:
            it = count_tokens(msgs[0]['content'])
            ot = count_tokens(msgs[1]['content'])
            input_tokens_list.append(it)
            output_tokens_list.append(ot)

    if input_tokens_list:
        total_in = sum(input_tokens_list)
        total_out = sum(output_tokens_list)
        stats['token_stats'] = {
            'total_input_tokens': total_in,
            'total_output_tokens': total_out,
            'total_tokens': total_in + total_out,
            'avg_input_tokens': total_in / len(input_tokens_list),
            'avg_output_tokens': total_out / len(output_tokens_list),
            'min_input_tokens': min(input_tokens_list),
            'max_input_tokens': max(input_tokens_list),
            'min_output_tokens': min(output_tokens_list),
            'max_output_tokens': max(output_tokens_list),
        }
    else:
        stats['token_stats'] = {
            'total_input_tokens': 0, 'total_output_tokens': 0,
            'total_tokens': 0, 'avg_input_tokens': 0, 'avg_output_tokens': 0,
            'min_input_tokens': 0, 'max_input_tokens': 0,
            'min_output_tokens': 0, 'max_output_tokens': 0,
        }

    return stats


def print_filter_stats(filter_stats: Dict[str, int], label: str = ""):
    """Print filter statistics."""
    total = filter_stats['total_input']
    if total == 0:
        print(f"\n  [{label}] No input records.")
        return

    passed = filter_stats['passed']
    rate = passed / total * 100

    print(f"\n{'=' * 70}")
    print(f"🔍 Filter Statistics — {label}")
    print(f"{'=' * 70}")
    print(f"   Total input records:                {total:,}")
    print(f"   Passed all filters:                 {passed:,}  ({rate:.1f}%)")
    print(f"   ---")
    print(f"   Filtered (no FIM response):         {filter_stats['filtered_no_fim']:,}")
    print(f"   Filtered (FIM parse failed):        {filter_stats['filtered_fim_parse_failed']:,}")
    print(f"   Filtered (no critique):             {filter_stats['filtered_no_critique']:,}")
    print(f"   Filtered (critique parse failed):   {filter_stats['filtered_critique_parse_failed']:,}")
    print(f"   Filtered (Gemini says discard):     {filter_stats['filtered_should_discard']:,}")
    print(f"   Filtered (per-func executability):  {filter_stats['filtered_low_per_func_executability']:,}")
    print(f"   Filtered (per-func score low):      {filter_stats['filtered_low_per_func_score']:,}")
    print(f"   Filtered (group overall low):       {filter_stats['filtered_low_group_overall']:,}")
    print(f"   Filtered (coherence low):           {filter_stats['filtered_low_coherence']:,}")


def print_data_stats(stats: Dict[str, Any]):
    """Print data statistics."""
    label = stats.get('label', '')
    print(f"\n{'=' * 70}")
    print(f"📊 Data Statistics — {label}")
    print(f"{'=' * 70}")
    print(f"   Total SFT samples: {stats['total_samples']:,}")

    # Group type breakdown
    gt_counts = stats['group_type_counts']
    if gt_counts:
        print(f"\n   📋 Group Type Breakdown:")
        for gt, cnt in sorted(gt_counts.items(), key=lambda x: -x[1]):
            print(f"      {gt:25s}  {cnt:,}")

    # Group overall score distribution
    g_dist = stats['group_overall_distribution']
    if g_dist:
        total = sum(g_dist.values())
        print(f"\n   📈 Group Overall Score Distribution:")
        for s in sorted(g_dist.keys()):
            cnt = g_dist[s]
            pct = cnt / total * 100 if total else 0
            bar = "█" * int(pct / 5)
            print(f"      Score {s}: {cnt:5,} ({pct:5.1f}%) {bar}")

    # Coherence score distribution
    c_dist = stats['coherence_distribution']
    if c_dist:
        total = sum(c_dist.values())
        print(f"\n   🔗 Group Coherence Score Distribution:")
        for s in sorted(c_dist.keys()):
            cnt = c_dist[s]
            pct = cnt / total * 100 if total else 0
            bar = "█" * int(pct / 5)
            print(f"      Score {s}: {cnt:5,} ({pct:5.1f}%) {bar}")

    # Per-function score distributions
    pf_dists = stats['per_func_score_distributions']
    print(f"\n   📝 Per-Function Score Distributions (across all functions in passed groups):")
    for score_name in ['correctness', 'executability', 'api_usage', 'completeness', 'per_func_overall']:
        dist = pf_dists.get(score_name, {})
        if not dist:
            continue
        total = sum(dist.values())
        print(f"\n      {score_name}:")
        for s in sorted(dist.keys()):
            cnt = dist[s]
            pct = cnt / total * 100 if total else 0
            bar = "█" * int(pct / 5)
            print(f"         Score {s}: {cnt:5,} ({pct:5.1f}%) {bar}")

    # Token stats
    ts = stats.get('token_stats', {})
    if ts and ts.get('total_tokens', 0) > 0:
        print(f"\n   📝 Token Statistics:")
        print(f"      Total input tokens:            {ts['total_input_tokens']:,}")
        print(f"      Total output tokens:           {ts['total_output_tokens']:,}")
        print(f"      Total tokens:                  {ts['total_tokens']:,}")
        print(f"      Avg input tokens per sample:   {ts['avg_input_tokens']:.1f}")
        print(f"      Avg output tokens per sample:  {ts['avg_output_tokens']:.1f}")
        print(f"      Input token range:             [{ts['min_input_tokens']:,}, {ts['max_input_tokens']:,}]")
        print(f"      Output token range:            [{ts['min_output_tokens']:,}, {ts['max_output_tokens']:,}]")

    print(f"\n{'=' * 70}")


# =============================================================================
# Save
# =============================================================================

def save_sft_data(sft_samples: List[Dict], output_path: str):
    """Save SFT training data as JSONL."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        for sample in sft_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    print(f"💾 Saved {len(sft_samples):,} samples → {output_path}")


# =============================================================================
# Pipeline for one group size
# =============================================================================

def process_group_size(
    checkpoint_dir: str,
    pattern: str,
    output_path: str,
    label: str,
    min_per_func_score: int,
    min_group_overall_score: int,
    min_coherence_score: int,
    save_stats_path: Optional[str] = None,
) -> Tuple[List[Dict], Dict, Dict]:
    """
    Full pipeline for one group size (pairs or triples):
    load → filter → construct SFT → stats → save.

    Returns:
        (sft_samples, filter_stats, data_stats)
    """
    print(f"\n{'#' * 70}")
    print(f"#  Processing: {label}")
    print(f"{'#' * 70}")

    # Step 1: Load
    print(f"\n📂 Step 1: Loading shard files...")
    records = load_shard_files(checkpoint_dir, pattern)
    if not records:
        print(f"   ❌ No records found for {label}. Skipping.")
        empty_stats = {'total_input': 0, 'passed': 0}
        return [], empty_stats, {}

    # Step 2: Filter
    print(f"🔍 Step 2: Filtering (per_func>={min_per_func_score}, "
          f"group_overall>={min_group_overall_score}, "
          f"coherence>={min_coherence_score})...")
    filtered, filter_stats = filter_multi_fim_samples(
        records,
        min_per_func_score=min_per_func_score,
        min_group_overall_score=min_group_overall_score,
        min_coherence_score=min_coherence_score,
    )
    print_filter_stats(filter_stats, label)

    # Step 3: Construct SFT samples
    print(f"\n🔧 Step 3: Constructing SFT samples...")
    sft_samples = []
    failed = 0
    for rec in filtered:
        sample = construct_multi_sft_sample(rec)
        if sample:
            sft_samples.append(sample)
        else:
            failed += 1
    print(f"   ✅ {len(sft_samples):,} SFT samples constructed")
    if failed:
        print(f"   ⚠️  {failed} failed to construct")

    # Step 4: Statistics
    print(f"\n📊 Step 4: Computing statistics...")
    data_stats = compute_statistics(filtered, sft_samples, label=label)
    print_data_stats(data_stats)

    # Step 5: Save
    if sft_samples:
        print(f"\n💾 Step 5: Saving...")
        save_sft_data(sft_samples, output_path)

    # Optional: save stats JSON
    if save_stats_path and sft_samples:
        stats_out = {
            'filter_stats': filter_stats,
            'data_stats': {
                'total_samples': data_stats['total_samples'],
                'group_type_counts': dict(data_stats['group_type_counts']),
                'group_overall_distribution': {
                    str(k): v for k, v in data_stats['group_overall_distribution'].items()
                },
                'coherence_distribution': {
                    str(k): v for k, v in data_stats['coherence_distribution'].items()
                },
                'token_stats': data_stats['token_stats'],
            }
        }
        with open(save_stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats_out, f, ensure_ascii=False, indent=2)
        print(f"📄 Stats saved → {save_stats_path}")

    return sft_samples, filter_stats, data_stats


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Prepare multi-function SFT training data from shard checkpoint files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python prepare_multi_sft_data.py \\
    -d /data/yubo/datasets/process_data_output_0227/ \\
    --pairs-pattern "step_4_multi_fim_output_0227_shard*_of_*_pairs.json" \\
    --triples-pattern "step_4_multi_fim_output_0227_shard*_of_*_triples.json" \\
    --output-pairs /data/yubo/datasets/multi_fim_sft_pairs_0227.jsonl \\
    --output-triples /data/yubo/datasets/multi_fim_sft_triples_0227.jsonl

  # Custom thresholds:
  python prepare_multi_sft_data.py -d /data/... \\
    --min-per-func-score 3 --min-group-overall-score 4 --min-coherence-score 3
        '''
    )

    # I/O
    parser.add_argument(
        "--checkpoint-dir", "-d",
        default="/data/yubo/datasets/process_data_output_0316/",
        help="Directory containing shard output files"
    )
    parser.add_argument(
        "--pairs-pattern",
        default="step_4_multi_fim_output_guided_0316_shard*_of_*_pairs.json",
        help="Glob pattern for pair shard files"
    )
    parser.add_argument(
        "--triples-pattern",
        default="step_4_multi_fim_output_guided_0316_shard*_of_*_triples.json",
        help="Glob pattern for triple shard files"
    )
    parser.add_argument(
        "--output-pairs",
        default="/data/yubo/datasets/hf_multi_functions_guided_fim_sft_0317/multi_functions_fim_sft_pairs_0317.jsonl",
        help="Output JSONL for pair SFT data"
    )
    parser.add_argument(
        "--output-triples",
        default="/data/yubo/datasets/hf_multi_functions_guided_fim_sft_0317/multi_functions_fim_sft_triples_0317.jsonl",
        help="Output JSONL for triple SFT data"
    )

    # Quality thresholds
    parser.add_argument("--min-per-func-score", type=int, default=3,
                        help="Min per-function overall_score (default: 3)")
    parser.add_argument("--min-group-overall-score", type=int, default=4,
                        help="Min group overall_score (default: 4)")
    parser.add_argument("--min-coherence-score", type=int, default=3,
                        help="Min group coherence_score (default: 3)")

    # Optional
    parser.add_argument("--save-stats", type=str, default="/data/yubo/datasets/hf_multi_functions_fim_sft_pairs_0227/stats.json",
                        help="Base path for stats JSON (appends _pairs/_triples)")

    args = parser.parse_args()

    print("🚀 Multi-Function SFT Training Data Preparation")
    print("=" * 70)
    print(f"   Checkpoint dir:       {args.checkpoint_dir}")
    print(f"   Pairs pattern:        {args.pairs_pattern}")
    print(f"   Triples pattern:      {args.triples_pattern}")
    print(f"   Min per-func score:   {args.min_per_func_score}")
    print(f"   Min group overall:    {args.min_group_overall_score}")
    print(f"   Min coherence:        {args.min_coherence_score}")

    # Process pairs
    pairs_stats_path = None
    if args.save_stats:
        base = Path(args.save_stats)
        pairs_stats_path = str(base.parent / f"{base.stem}_pairs{base.suffix}")

    pair_samples, pair_fstats, pair_dstats = process_group_size(
        checkpoint_dir=args.checkpoint_dir,
        pattern=args.pairs_pattern,
        output_path=args.output_pairs,
        label="Pairs (2 functions)",
        min_per_func_score=args.min_per_func_score,
        min_group_overall_score=args.min_group_overall_score,
        min_coherence_score=args.min_coherence_score,
        save_stats_path=pairs_stats_path,
    )

    # Process triples
    triples_stats_path = None
    if args.save_stats:
        base = Path(args.save_stats)
        triples_stats_path = str(base.parent / f"{base.stem}_triples{base.suffix}")

    triple_samples, triple_fstats, triple_dstats = process_group_size(
        checkpoint_dir=args.checkpoint_dir,
        pattern=args.triples_pattern,
        output_path=args.output_triples,
        label="Triples (3 functions)",
        min_per_func_score=args.min_per_func_score,
        min_group_overall_score=args.min_group_overall_score,
        min_coherence_score=args.min_coherence_score,
        save_stats_path=triples_stats_path,
    )

    # Final summary
    print(f"\n{'=' * 70}")
    print(f"✅ FINAL SUMMARY")
    print(f"{'=' * 70}")
    print(f"   Pairs:   {len(pair_samples):,} SFT samples "
          f"(from {pair_fstats.get('total_input', 0):,} input records)")
    print(f"   Triples: {len(triple_samples):,} SFT samples "
          f"(from {triple_fstats.get('total_input', 0):,} input records)")
    print(f"   Total:   {len(pair_samples) + len(triple_samples):,} SFT samples")

    if pair_dstats.get('token_stats', {}).get('total_tokens', 0) > 0:
        pt = pair_dstats['token_stats']['total_tokens']
        tt = triple_dstats.get('token_stats', {}).get('total_tokens', 0)
        print(f"   Total tokens (pairs):   {pt:,}")
        print(f"   Total tokens (triples): {tt:,}")
        print(f"   Total tokens (all):     {pt + tt:,}")

    print(f"\n   Output files:")
    print(f"     Pairs:   {args.output_pairs}")
    print(f"     Triples: {args.output_triples}")
    print(f"{'=' * 70}")
    print("✅ Done!")


if __name__ == "__main__":
    main()