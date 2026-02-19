#!/usr/bin/env python3
"""
SFT Training Data Preparation Script (adapted for shard-based checkpoint files)

Processes shard checkpoint files and generates SFT training data.
Filters low-quality samples and formats them for instruction fine-tuning.
"""

import os
import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Optional
import tiktoken

# =============================================================================
# Prompt Template (same as step 4 FIM completion prompt)
# =============================================================================

FIM_COMPLETION_PROMPT = '''You are an expert Python programmer. Your task is to complete a masked function based on the surrounding code context.

## Task

Below is a Python file where one function's body has been replaced with `# <MASKED_FUNCTION_BODY>`. Your job is to:

1. **Analyze the context**: Look at the imports, other functions, class definitions, and how this function is called or calls other functions.
2. **Reason step by step**: Think through what this function should do based on:
   - The function signature (name, parameters, type hints)
   - How it's used elsewhere in the code
   - What other functions it might need to call
   - The overall purpose of the code
3. **Write the complete function body**: Provide a working implementation.

## Output Format

Please structure your response as follows:

### Reasoning
<Your step-by-step analysis of what the function should do>

### Implementation
```python
<The complete function body code only, without the function signature>
```

## Important Notes
- Only provide the function BODY (the code inside the function), not the signature
- Maintain proper indentation (the body should be indented with 4 spaces or appropriate level)
- The implementation should be consistent with the coding style in the file
- If the function has a docstring, include it as part of the body

## Code with Masked Function

```python
{masked_code}
```

## Function to Complete

Function name: `{function_name}`

Please analyze the context and provide your reasoning followed by the implementation.
'''


# Special tokens to remove from training data
SPECIAL_TOKENS_TO_REMOVE = [
    '<|endoftext|>',
    '<|fim_prefix|>',
    '<|fim_middle|>',
    '<|fim_suffix|>',
    '<|fim▁begin|>',
    '<|fim▁hole|>',
    '<|fim▁end|>',
    '<|pad|>',
    '<|eos|>',
    '<|bos|>',
    '<|sep|>',
    '<|cls|>',
    '<|mask|>',
]


def clean_special_tokens(text: str) -> str:
    """Remove special tokens from text that could interfere with training."""
    cleaned = text
    for token in SPECIAL_TOKENS_TO_REMOVE:
        cleaned = cleaned.replace(token, '')
    return cleaned


def load_shard_files(checkpoint_dir: str, checkpoint_pattern: str) -> list[dict]:
    """
    Load all shard checkpoint files from a directory.

    Args:
        checkpoint_dir: Directory containing shard files
        checkpoint_pattern: Glob pattern for shard files

    Returns:
        List of all records from all shard files
    """
    checkpoint_path = Path(checkpoint_dir)
    all_records = []
    loaded_files = 0

    shard_files = sorted(checkpoint_path.glob(checkpoint_pattern))

    if not shard_files:
        print(f"⚠️ No files found matching '{checkpoint_pattern}' in {checkpoint_dir}")
        return all_records

    print(f"📂 Found {len(shard_files)} shard files")

    for shard_file in shard_files:
        try:
            with open(shard_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Shard files have 'results' key containing actual records
            results = data.get('results', [])
            all_records.extend(results)
            loaded_files += 1
            print(f"   ✅ Loaded {len(results):>4d} records from {shard_file.name}")

        except Exception as e:
            print(f"   ❌ Failed to load {shard_file.name}: {e}")

    print(f"\n📊 Total: Loaded {len(all_records):,} records from {loaded_files} files")
    return all_records


def filter_low_quality_samples(
        records: list[dict],
        min_individual_score: int = 3,
        min_overall_score: int = 4
) -> tuple[list[dict], dict]:
    """
    Filter out low-quality samples based on critique scores.

    Filtering criteria:
      - Individual dimension scores must be >= min_individual_score (default: >= 3)
      - Overall score must be >= min_overall_score (default: >= 4)
    """
    filtered_records = []
    filter_stats = {
        'total_input': len(records),
        'passed': 0,
        'filtered_no_critique': 0,
        'filtered_parse_failed': 0,
        'filtered_no_fim_implementation': 0,
        'filtered_low_individual_score': 0,
        'filtered_low_overall_score': 0,
    }

    individual_score_keys = ['correctness', 'executability', 'api_usage', 'readability', 'completeness']

    for record in records:
        # Check critique response exists
        critique_response = record.get('critique_response', {})
        if not critique_response:
            filter_stats['filtered_no_critique'] += 1
            continue

        # Check critique parsing succeeded
        if not critique_response.get('parse_success', False):
            filter_stats['filtered_parse_failed'] += 1
            continue

        # Check FIM implementation exists
        fim_response = record.get('fim_response', {})
        if not fim_response.get('parse_success', False) or not fim_response.get('implementation'):
            filter_stats['filtered_no_fim_implementation'] += 1
            continue

        # Check overall score >= min_overall_score
        overall_score = critique_response.get('overall_score', 0)
        if overall_score < min_overall_score:
            filter_stats['filtered_low_overall_score'] += 1
            continue

        # Check individual scores >= min_individual_score
        scores = critique_response.get('scores', {})
        low_score_found = False
        for key in individual_score_keys:
            score = scores.get(key, 0)
            if score < min_individual_score:
                low_score_found = True
                break

        if low_score_found:
            filter_stats['filtered_low_individual_score'] += 1
            continue

        # Passed all filters
        filtered_records.append(record)
        filter_stats['passed'] += 1

    return filtered_records, filter_stats


def construct_sft_sample(record: dict) -> Optional[dict]:
    """
    Construct a single SFT training sample from a record.

    Returns:
        SFT sample in messages format, or None if construction fails
    """
    try:
        # Use masked_code_content field (from shard format)
        masked_code = record.get('masked_code_content', '') or record.get('masked_code', '')
        function_name = record.get('func_name', '') or record.get('function_name', '')
        fim_response = record.get('fim_response', {})

        if not masked_code or not function_name:
            return None

        # Construct user message
        user_content = FIM_COMPLETION_PROMPT.format(
            masked_code=masked_code,
            function_name=function_name
        )

        # Construct assistant message (raw response with reasoning + implementation)
        assistant_content = fim_response.get('raw_response', '')
        if not assistant_content:
            return None

        # Clean special tokens
        user_content = clean_special_tokens(user_content)
        assistant_content = clean_special_tokens(assistant_content)

        return {
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content}
            ]
        }

    except Exception as e:
        print(f"   ⚠️ Failed to construct sample for {record.get('unique_id', 'unknown')}: {e}")
        return None


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Count tokens in text using tiktoken."""
    try:
        encoding = tiktoken.get_encoding(encoding_name)
        return len(encoding.encode(text, disallowed_special=()))
    except Exception:
        return len(text) // 4


def compute_statistics(records: list[dict], sft_samples: list[dict]) -> dict:
    """Compute statistics about the data."""
    stats = {
        'total_samples': len(sft_samples),
        'score_distributions': {},
        'token_stats': {
            'total_input_tokens': 0,
            'total_output_tokens': 0,
            'total_tokens': 0,
            'avg_input_tokens': 0,
            'avg_output_tokens': 0,
            'min_input_tokens': float('inf'),
            'max_input_tokens': 0,
            'min_output_tokens': float('inf'),
            'max_output_tokens': 0,
        }
    }

    # Score distributions
    score_keys = ['correctness', 'executability', 'api_usage', 'readability', 'completeness', 'overall_score']
    for key in score_keys:
        stats['score_distributions'][key] = defaultdict(int)

    for record in records:
        critique = record.get('critique_response', {})
        scores = critique.get('scores', {})
        for key in ['correctness', 'executability', 'api_usage', 'readability', 'completeness']:
            score = scores.get(key, 0)
            stats['score_distributions'][key][score] += 1
        overall = critique.get('overall_score', 0)
        stats['score_distributions']['overall_score'][overall] += 1

    # Token statistics
    input_tokens_list = []
    output_tokens_list = []

    for sample in sft_samples:
        messages = sample.get('messages', [])
        if len(messages) >= 2:
            input_tokens = count_tokens(messages[0].get('content', ''))
            output_tokens = count_tokens(messages[1].get('content', ''))
            input_tokens_list.append(input_tokens)
            output_tokens_list.append(output_tokens)
            stats['token_stats']['total_input_tokens'] += input_tokens
            stats['token_stats']['total_output_tokens'] += output_tokens

    if input_tokens_list:
        stats['token_stats']['total_tokens'] = (
            stats['token_stats']['total_input_tokens'] + stats['token_stats']['total_output_tokens']
        )
        stats['token_stats']['avg_input_tokens'] = sum(input_tokens_list) / len(input_tokens_list)
        stats['token_stats']['avg_output_tokens'] = sum(output_tokens_list) / len(output_tokens_list)
        stats['token_stats']['min_input_tokens'] = min(input_tokens_list)
        stats['token_stats']['max_input_tokens'] = max(input_tokens_list)
        stats['token_stats']['min_output_tokens'] = min(output_tokens_list)
        stats['token_stats']['max_output_tokens'] = max(output_tokens_list)

    return stats


def print_statistics(filter_stats: dict, data_stats: dict):
    """Print formatted statistics."""
    print("\n" + "=" * 70)
    print("📊 DATA STATISTICS")
    print("=" * 70)

    print("\n🔍 Filter Statistics:")
    print(f"   Total input records:           {filter_stats['total_input']:,}")
    print(f"   Passed all filters:            {filter_stats['passed']:,}")
    print(f"   Filtered (no critique):        {filter_stats['filtered_no_critique']:,}")
    print(f"   Filtered (parse failed):       {filter_stats['filtered_parse_failed']:,}")
    print(f"   Filtered (no FIM impl):        {filter_stats['filtered_no_fim_implementation']:,}")
    print(f"   Filtered (low individual <3):  {filter_stats['filtered_low_individual_score']:,}")
    print(f"   Filtered (low overall <4):     {filter_stats['filtered_low_overall_score']:,}")

    pass_rate = filter_stats['passed'] / filter_stats['total_input'] * 100 if filter_stats['total_input'] > 0 else 0
    print(f"   Pass rate:                     {pass_rate:.1f}%")

    print("\n📈 Score Distributions:")
    for score_name, dist in data_stats['score_distributions'].items():
        print(f"\n   {score_name}:")
        total = sum(dist.values())
        for score in sorted(dist.keys()):
            count = dist[score]
            pct = count / total * 100 if total > 0 else 0
            bar = "█" * int(pct / 5)
            print(f"      Score {score}: {count:5,} ({pct:5.1f}%) {bar}")

    print("\n📝 Token Statistics:")
    ts = data_stats['token_stats']
    print(f"   Total samples:                 {data_stats['total_samples']:,}")
    print(f"   Total input tokens:            {ts['total_input_tokens']:,}")
    print(f"   Total output tokens:           {ts['total_output_tokens']:,}")
    print(f"   Total tokens:                  {ts['total_tokens']:,}")
    print(f"   Avg input tokens per sample:   {ts['avg_input_tokens']:.1f}")
    print(f"   Avg output tokens per sample:  {ts['avg_output_tokens']:.1f}")
    print(f"   Input token range:             [{ts['min_input_tokens']:,}, {ts['max_input_tokens']:,}]")
    print(f"   Output token range:            [{ts['min_output_tokens']:,}, {ts['max_output_tokens']:,}]")
    print("\n" + "=" * 70)


def save_sft_data(sft_samples: list[dict], output_path: str):
    """Save SFT training data as JSONL."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in sft_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"\n💾 Saved {len(sft_samples)} samples to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare SFT training data from shard checkpoint files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Default usage with shard files:
  python prepare_sft_data_0219.py

  # Custom paths and thresholds:
  python prepare_sft_data_0219.py -d /path/to/shards -o output.jsonl \\
    --min-individual-score 3 --min-overall-score 4
        '''
    )

    parser.add_argument(
        "--checkpoint-dir", "-d",
        default="/data/yubo/datasets/temp_data_0219",
        help="Directory containing shard checkpoint files"
    )
    parser.add_argument(
        "--output", "-o",
        default="/data/yubo/datasets/process_data_output_0215/step_5_sft_data/fim_sft_data_temp_0219.jsonl",
        help="Output JSONL file path"
    )
    parser.add_argument(
        "--checkpoint-pattern", "-p",
        default="step_4_fim_critique_0217*.json",
        help="Glob pattern for shard files"
    )
    parser.add_argument(
        "--min-individual-score",
        type=int,
        default=3,
        help="Minimum individual dimension score (inclusive, default: 3)"
    )
    parser.add_argument(
        "--min-overall-score",
        type=int,
        default=4,
        help="Minimum overall score (inclusive, default: 4)"
    )
    parser.add_argument(
        "--save-stats",
        type=str,
        default=None,
        help="Path to save statistics as JSON (optional)"
    )

    args = parser.parse_args()

    print("🚀 SFT Training Data Preparation (Shard Mode)")
    print("=" * 70)

    # Step 1: Load shard files
    print("\n📂 Step 1: Loading shard files...")
    all_records = load_shard_files(args.checkpoint_dir, args.checkpoint_pattern)

    if not all_records:
        print("❌ No records found. Exiting.")
        return

    # Step 2: Filter low-quality samples
    print("\n🔍 Step 2: Filtering low-quality samples...")
    print(f"   Filter criteria:")
    print(f"   - Individual scores (correctness/executability/api_usage/readability/completeness) >= {args.min_individual_score}")
    print(f"   - Overall score >= {args.min_overall_score}")

    filtered_records, filter_stats = filter_low_quality_samples(
        all_records,
        min_individual_score=args.min_individual_score,
        min_overall_score=args.min_overall_score
    )

    print(f"   ✅ {len(filtered_records)} records passed filters")

    # Step 3: Construct SFT samples
    print("\n🔧 Step 3: Constructing SFT samples...")
    sft_samples = []
    failed_count = 0
    for record in filtered_records:
        sample = construct_sft_sample(record)
        if sample:
            sft_samples.append(sample)
        else:
            failed_count += 1

    print(f"   ✅ Constructed {len(sft_samples)} SFT samples")
    if failed_count:
        print(f"   ⚠️ {failed_count} samples failed to construct")

    # Step 4: Compute and print statistics
    print("\n📊 Step 4: Computing statistics...")
    data_stats = compute_statistics(filtered_records, sft_samples)
    print_statistics(filter_stats, data_stats)

    # Step 5: Save SFT data
    print("\n💾 Step 5: Saving SFT training data...")
    save_sft_data(sft_samples, args.output)

    # Optionally save statistics
    if args.save_stats:
        stats_output = {
            'filter_stats': filter_stats,
            'data_stats': {
                'total_samples': data_stats['total_samples'],
                'token_stats': data_stats['token_stats'],
                'score_distributions': {
                    k: dict(v) for k, v in data_stats['score_distributions'].items()
                }
            }
        }
        with open(args.save_stats, 'w', encoding='utf-8') as f:
            json.dump(stats_output, f, ensure_ascii=False, indent=2)
        print(f"📄 Statistics saved to {args.save_stats}")

    print("\n✅ Done!")


if __name__ == "__main__":
    main()