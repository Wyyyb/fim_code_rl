#!/usr/bin/env python3
"""
step_2_extract_python_files_1223.py

Extract Python files from downloaded repositories and save to JSON format.
Each Python file becomes one record with code content and metadata.
"""

import argparse
import ast
import csv
import json
import os
import sys
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse


def parse_repo_url(url: str) -> str:
    """Extract repository name from GitHub URL."""
    parsed = urlparse(url.strip())
    path_parts = parsed.path.strip('/').split('/')
    if len(path_parts) >= 2:
        return path_parts[-1]
    return None


def load_repo_metadata(csv_file: Path) -> dict:
    """Load repository metadata from CSV file into a dict keyed by repo_id."""
    metadata = {}

    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            repo_id = row.get('repo_id', '').strip()
            if repo_id:
                metadata[repo_id] = {
                    'repo_id': repo_id,
                    'repository_url': row.get('repository_url', '').strip(),
                    'category': row.get('category', '').strip(),
                    'quality_rating': row.get('quality_rating', '').strip(),
                    'description': row.get('description', '').strip(),
                    'stars_estimate': row.get('stars_estimate', '').strip(),
                    'notes': row.get('notes', '').strip(),
                }

    return metadata


def find_repo_folders(repos_dir: Path) -> dict:
    """
    Find all repo folders and map repo_id to folder path.
    Folder naming convention: {repo_id}_{repo_name}
    """
    repo_folders = {}

    for folder in repos_dir.iterdir():
        if folder.is_dir():
            folder_name = folder.name
            # Extract repo_id (everything before the first underscore)
            if '_' in folder_name:
                repo_id = folder_name.split('_')[0]
                repo_folders[repo_id] = folder

    return repo_folders


def read_python_file(file_path: Path) -> Optional[str]:
    """Read Python file content, handling encoding issues."""
    encodings = ['utf-8', 'latin-1', 'cp1252']

    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"  [WARN] Error reading {file_path}: {e}")
            return None

    print(f"  [WARN] Could not decode {file_path} with any encoding")
    return None


def count_lines(content: str) -> int:
    """Count the number of lines in the code content."""
    if not content:
        return 0
    return len(content.splitlines())


def count_functions(content: str) -> int:
    """
    Count the number of functions in the code.
    Includes top-level functions and class methods.
    Does not count classes themselves.
    """
    if not content:
        return 0

    try:
        tree = ast.parse(content)
    except SyntaxError:
        # If parsing fails, return -1 to indicate error
        return -1

    function_count = 0

    for node in ast.walk(tree):
        # Count FunctionDef (regular functions and methods)
        # Count AsyncFunctionDef (async functions and methods)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            function_count += 1

    return function_count


def extract_python_files(repo_folder: Path, repo_metadata: dict, global_id_counter: int) -> tuple[list, int]:
    """
    Extract all Python files from a repository folder.
    Returns (records, updated_counter).
    """
    records = []
    current_id = global_id_counter

    # Walk through all files in the repo
    for root, dirs, files in os.walk(repo_folder):
        # Skip hidden directories and common non-source directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in [
            '__pycache__', 'node_modules', 'venv', 'env', '.git',
            'build', 'dist', 'egg-info', '.eggs', '.tox'
        ]]

        for file in files:
            if file.endswith('.py'):
                file_path = Path(root) / file

                # Get relative path from repo folder
                relative_path = file_path.relative_to(repo_folder)
                # Format as /path/to/file.py
                relative_path_str = '/' + str(relative_path).replace('\\', '/')

                # Read file content
                content = read_python_file(file_path)
                if content is None:
                    continue

                # Count lines
                line_num = count_lines(content)

                # Count functions
                func_num = count_functions(content)

                # Create record
                record = {
                    'sample_id': current_id,
                    'repo_id': repo_metadata['repo_id'],
                    'repository_url': repo_metadata['repository_url'],
                    'file_path': relative_path_str,
                    'line_num': line_num,
                    'func_num': func_num,
                    'category': repo_metadata['category'],
                    'quality_rating': repo_metadata['quality_rating'],
                    'description': repo_metadata['description'],
                    'stars_estimate': repo_metadata['stars_estimate'],
                    'notes': repo_metadata['notes'],
                    'code_content': content,
                }

                records.append(record)
                current_id += 1

    return records, current_id


def main():
    parser = argparse.ArgumentParser(
        description='Extract Python files from downloaded repos to JSON format.'
    )
    parser.add_argument(
        '--csv_file',
        type=str,
        default='code_repos_csv_1223.txt',
        help='Path to the CSV file with repo metadata'
    )
    parser.add_argument(
        '--repos_dir',
        type=str,
        default='/data/yubo/datasets/collected_sc_1223',
        help='Path to the directory containing downloaded repos'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default='extracted_python_files_1223.json',
        help='Path for the output JSON file'
    )

    args = parser.parse_args()

    csv_file = Path(args.csv_file)
    repos_dir = Path(args.repos_dir)
    output_path = Path(args.output_path)

    # Validate inputs
    if not csv_file.exists():
        print(f"Error: CSV file not found: {csv_file}")
        sys.exit(1)

    if not repos_dir.exists():
        print(f"Error: Repos directory not found: {repos_dir}")
        sys.exit(1)

    print(f"CSV file: {csv_file}")
    print(f"Repos directory: {repos_dir}")
    print(f"Output path: {output_path}")
    print()

    # Load metadata
    print("Loading repository metadata...")
    repo_metadata = load_repo_metadata(csv_file)
    print(f"  Found {len(repo_metadata)} repos in CSV")

    # Find repo folders
    print("Scanning repository folders...")
    repo_folders = find_repo_folders(repos_dir)
    print(f"  Found {len(repo_folders)} repo folders")
    print()

    # Extract Python files
    all_records = []
    repos_processed = 0
    repos_not_found = 0
    global_id_counter = 0
    parse_error_count = 0

    for repo_id, metadata in repo_metadata.items():
        if repo_id not in repo_folders:
            print(f"[{repo_id}] Folder not found, skipping")
            repos_not_found += 1
            continue

        repo_folder = repo_folders[repo_id]
        print(f"[{repo_id}] Processing: {repo_folder.name}")

        records, global_id_counter = extract_python_files(repo_folder, metadata, global_id_counter)
        all_records.extend(records)

        # Count parse errors in this batch
        batch_errors = sum(1 for r in records if r['func_num'] == -1)
        parse_error_count += batch_errors

        print(f"  Extracted {len(records)} Python files" + (
            f" ({batch_errors} parse errors)" if batch_errors > 0 else ""))
        repos_processed += 1

    # Save to JSON
    print()
    print(f"Writing {len(all_records)} records to {output_path}...")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)

    # Summary
    print()
    print("=" * 50)
    print("Extraction Summary:")
    print(f"  Repos in CSV:       {len(repo_metadata)}")
    print(f"  Repos processed:    {repos_processed}")
    print(f"  Repos not found:    {repos_not_found}")
    print(f"  Python files:       {len(all_records)}")
    print(f"  Parse errors:       {parse_error_count} (func_num = -1)")
    print(f"  Output file:        {output_path}")
    print("=" * 50)


if __name__ == "__main__":
    '''
    python step_2_extract_python_files_1223.py \
    --csv_file code_repos_csv_1223.txt \
    --repos_dir /data/yubo/datasets/collected_sc_1223 \
    --output_path /data/yubo/datasets/extracted_python_files_1223.json
    '''
    main()