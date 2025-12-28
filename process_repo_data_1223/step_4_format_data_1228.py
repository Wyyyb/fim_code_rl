#!/usr/bin/env python3
"""
Step 4: Format FIM Training Data

This script reads the output from step 3 (Gemini analysis results) and formats it
into a flat structure where each selected function becomes a separate data entry.

Input: step_3_fim_training_data.json (nested structure with selected_function_list)
Output: step_4_format_fim_query_data.json (flat structure, one entry per function)
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_json_data(file_path: str) -> List[Dict[str, Any]]:
    """Load JSON data from file."""
    logger.info(f"Loading data from: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} samples")
    return data


def save_json_data(data: List[Dict[str, Any]], file_path: str):
    """Save JSON data to file."""
    # Ensure output directory exists
    output_path = Path(file_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving {len(data)} entries to: {file_path}")
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info("Data saved successfully")


def format_fim_data(input_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Format FIM training data.

    Transforms nested structure (samples with selected_function_list) into
    flat structure (one entry per selected function).

    Args:
        input_data: List of samples from step 3 output

    Returns:
        List of formatted entries, one per selected function
    """
    formatted_data = []

    # Statistics
    total_samples = len(input_data)
    samples_with_functions = 0
    samples_without_key = 0
    samples_with_empty_list = 0
    total_functions = 0

    for sample in input_data:
        # Check if selected_function_list key exists
        if 'selected_function_list' not in sample:
            samples_without_key += 1
            continue

        selected_functions = sample['selected_function_list']

        # Check if the list is empty
        if not selected_functions:
            samples_with_empty_list += 1
            continue

        samples_with_functions += 1

        # Extract common sample information
        sample_info = {
            'sample_id': sample.get('sample_id'),
            'repo_id': sample.get('repo_id'),
            'file_path': sample.get('file_path'),
            'code_content': sample.get('code_content'),
            'line_num': sample.get('line_num'),
            'func_num': sample.get('func_num'),
        }

        # Create one entry per selected function
        for func in selected_functions:
            entry = {
                # Sample-level information
                **sample_info,
                # Function-level information
                'function_id': func.get('function_id'),
                'function_name': func.get('function_name'),
                'function_code': func.get('function_code'),
                'masked_code': func.get('masked_code'),
                'start_line': func.get('start_line'),
                'end_line': func.get('end_line'),
                'difficulty_score': func.get('difficulty_score'),
                'selection_reason': func.get('selection_reason'),
            }
            formatted_data.append(entry)
            total_functions += 1

    # Print statistics
    logger.info("=" * 60)
    logger.info("Formatting Statistics:")
    logger.info(f"  Total input samples: {total_samples}")
    logger.info(f"  Samples without 'selected_function_list' key: {samples_without_key}")
    logger.info(f"  Samples with empty function list: {samples_with_empty_list}")
    logger.info(f"  Samples with selected functions: {samples_with_functions}")
    logger.info(f"  Total functions extracted: {total_functions}")
    logger.info("=" * 60)

    return formatted_data


def print_sample_preview(data: List[Dict[str, Any]], num_samples: int = 2):
    """Print a preview of the formatted data."""
    if not data:
        logger.info("No data to preview")
        return

    print("\n" + "=" * 60)
    print("Sample Preview (first {} entries):".format(min(num_samples, len(data))))
    print("=" * 60)

    for i, entry in enumerate(data[:num_samples]):
        print(f"\n--- Entry {i + 1} ---")
        print(f"  sample_id: {entry.get('sample_id')}")
        print(f"  repo_id: {entry.get('repo_id')}")
        print(f"  file_path: {entry.get('file_path')}")
        print(f"  line_num: {entry.get('line_num')}")
        print(f"  func_num: {entry.get('func_num')}")
        print(f"  function_id: {entry.get('function_id')}")
        print(f"  function_name: {entry.get('function_name')}")
        print(f"  difficulty_score: {entry.get('difficulty_score')}")
        print(f"  selection_reason: {entry.get('selection_reason', '')[:100]}...")
        print(f"  function_code length: {len(entry.get('function_code', ''))} chars")
        print(f"  masked_code length: {len(entry.get('masked_code', ''))} chars")
        print(f"  start_line: {entry.get('start_line')}, end_line: {entry.get('end_line')}")

    print("\n" + "=" * 60)


def print_statistics(data: List[Dict[str, Any]]):
    """Print detailed statistics about the formatted data."""
    if not data:
        return

    print("\n" + "=" * 60)
    print("Detailed Statistics:")
    print("=" * 60)

    # Difficulty score distribution
    difficulty_counts = {}
    for entry in data:
        score = entry.get('difficulty_score', 'N/A')
        difficulty_counts[score] = difficulty_counts.get(score, 0) + 1

    print("\nDifficulty Score Distribution:")
    for score in sorted(difficulty_counts.keys(),
                        key=lambda x: (isinstance(x, str), x if isinstance(x, (int, float)) else 0)):
        count = difficulty_counts[score]
        percentage = count / len(data) * 100
        print(f"  Score {score}: {count} functions ({percentage:.1f}%)")

    # Functions per sample
    sample_func_counts = {}
    for entry in data:
        sample_id = entry.get('sample_id')
        sample_func_counts[sample_id] = sample_func_counts.get(sample_id, 0) + 1

    if sample_func_counts:
        avg_funcs = sum(sample_func_counts.values()) / len(sample_func_counts)
        max_funcs = max(sample_func_counts.values())
        min_funcs = min(sample_func_counts.values())
        print(f"\nFunctions per Sample:")
        print(f"  Average: {avg_funcs:.2f}")
        print(f"  Min: {min_funcs}")
        print(f"  Max: {max_funcs}")
        print(f"  Unique samples: {len(sample_func_counts)}")

    # Function code length statistics
    func_lengths = [len(entry.get('function_code', '')) for entry in data]
    if func_lengths:
        avg_length = sum(func_lengths) / len(func_lengths)
        print(f"\nFunction Code Length (chars):")
        print(f"  Average: {avg_length:.0f}")
        print(f"  Min: {min(func_lengths)}")
        print(f"  Max: {max(func_lengths)}")

    # Function line count
    func_line_counts = [
        entry.get('end_line', 0) - entry.get('start_line', 0)
        for entry in data
        if entry.get('end_line') is not None and entry.get('start_line') is not None
    ]
    if func_line_counts:
        avg_lines = sum(func_line_counts) / len(func_line_counts)
        print(f"\nFunction Line Count:")
        print(f"  Average: {avg_lines:.1f}")
        print(f"  Min: {min(func_line_counts)}")
        print(f"  Max: {max(func_line_counts)}")

    print("\n" + "=" * 60)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Format FIM training data from step 3 output"
    )
    parser.add_argument(
        "--input", "-i",
        default="/data/yubo/datasets/process_data_output_1228/step_3_fim_training_data.json",
        help="Path to input JSON file (step 3 output)"
    )
    parser.add_argument(
        "--output", "-o",
        default="/data/yubo/datasets/process_data_output_1228/step_4_format_fim_query_data.json",
        help="Path to output JSON file"
    )
    parser.add_argument(
        "--preview", "-p",
        type=int,
        default=2,
        help="Number of sample entries to preview (default: 2)"
    )
    parser.add_argument(
        "--no-stats",
        action="store_true",
        help="Disable detailed statistics output"
    )

    args = parser.parse_args()

    # Check if input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {args.input}")
        return 1

    # Load data
    input_data = load_json_data(args.input)

    # Format data
    formatted_data = format_fim_data(input_data)

    if not formatted_data:
        logger.warning("No data to save after formatting")
        return 1

    # Save formatted data
    save_json_data(formatted_data, args.output)

    # Print preview
    if args.preview > 0:
        print_sample_preview(formatted_data, args.preview)

    # Print statistics
    if not args.no_stats:
        print_statistics(formatted_data)

    print(f"\n✅ Successfully formatted {len(formatted_data)} function entries")
    print(f"   Output saved to: {args.output}")

    return 0


if __name__ == "__main__":
    """
    Usage:
        python step_4_format_data_1228.py
        python step_4_format_data_1228.py --input /path/to/input.json --output /path/to/output.json
        python step_4_format_data_1228.py --preview 5  # Preview more samples
        python step_4_format_data_1228.py --no-stats   # Skip detailed statistics
    """
    exit(main())