#!/usr/bin/env python3
"""
step_1_download_code_repo_1223.py

Download code repositories from the CSV file using git clone.
Each repository is saved with the naming format: {repo_id}_{repo_name}
"""

import csv
import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse


def parse_repo_url(url: str) -> str:
    """Extract repository name from GitHub URL."""
    parsed = urlparse(url.strip())
    path_parts = parsed.path.strip('/').split('/')
    if len(path_parts) >= 2:
        return path_parts[-1]  # repo name is the last part
    return None


def clone_repo(repo_id: str, repo_url: str, target_dir: Path) -> bool:
    """Clone a single repository to the target directory."""
    repo_name = parse_repo_url(repo_url)
    if not repo_name:
        print(f"  [ERROR] Cannot parse repo name from URL: {repo_url}")
        return False

    # Create folder name: {repo_id}_{repo_name}
    folder_name = f"{repo_id}_{repo_name}"
    target_path = target_dir / folder_name

    if target_path.exists():
        print(f"  [SKIP] Already exists: {folder_name}")
        return True

    try:
        print(f"  [CLONE] {repo_url} -> {folder_name}")
        result = subprocess.run(
            ["git", "clone", "--depth", "1", repo_url.strip(), str(target_path)],
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes timeout
        )

        if result.returncode == 0:
            print(f"  [OK] Successfully cloned {folder_name}")
            return True
        else:
            print(f"  [ERROR] Failed to clone {folder_name}: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print(f"  [ERROR] Timeout while cloning {folder_name}")
        return False
    except Exception as e:
        print(f"  [ERROR] Exception while cloning {folder_name}: {e}")
        return False


def main():
    # Configuration
    csv_file = Path("code_repos_csv_1223.txt")
    target_dir = Path("/data/yubo/datasets/collected_sc_1223")

    # Check if CSV file exists
    if not csv_file.exists():
        print(f"Error: CSV file not found: {csv_file}")
        sys.exit(1)

    # Create target directory
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"Target directory: {target_dir.absolute()}")

    # Read CSV and clone repos
    success_count = 0
    fail_count = 0
    skip_count = 0

    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            repo_id = row.get('repo_id', '').strip()
            repo_url = row.get('repository_url', '').strip()

            if not repo_id or not repo_url:
                print(f"  [SKIP] Invalid row: {row}")
                skip_count += 1
                continue

            print(f"\n[{repo_id}] Processing: {repo_url}")

            if clone_repo(repo_id, repo_url, target_dir):
                success_count += 1
            else:
                fail_count += 1

    # Summary
    print("\n" + "=" * 50)
    print("Download Summary:")
    print(f"  Success: {success_count}")
    print(f"  Failed:  {fail_count}")
    print(f"  Skipped: {skip_count}")
    print(f"  Total:   {success_count + fail_count + skip_count}")
    print("=" * 50)


if __name__ == "__main__":
    main()