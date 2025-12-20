import os
import ast
import sys
from pathlib import Path
from typing import Set, Dict, List, Tuple
from collections import defaultdict, OrderedDict


class ImportAnalyzer:
    """分析Python文件的import依赖关系"""

    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.py_files = self._find_all_py_files()
        self.module_map = self._build_module_map()

    def _find_all_py_files(self) -> Dict[str, Path]:
        """找到所有Python文件并建立映射"""
        py_files = {}
        for py_file in self.repo_path.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
            rel_path = py_file.relative_to(self.repo_path)
            module_name = str(rel_path.with_suffix("")).replace(os.sep, ".")
            py_files[module_name] = py_file
        return py_files

    def _build_module_map(self) -> Dict[str, str]:
        """建立模块名到文件路径的映射"""
        module_map = {}
        for module_name, file_path in self.py_files.items():
            parts = module_name.split(".")
            for i in range(len(parts)):
                partial_name = ".".join(parts[i:])
                if partial_name not in module_map:
                    module_map[partial_name] = module_name
        return module_map

    def parse_imports(self, file_path: Path) -> Tuple[List[str], List[str]]:
        """解析文件的import语句

        Returns:
            (internal_imports, external_imports)
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            tree = ast.parse(content)
        except Exception as e:
            print(f"Warning: Failed to parse {file_path}: {e}")
            return [], []

        internal_imports = []
        external_imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self._categorize_import(alias.name, internal_imports, external_imports)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    self._categorize_import(node.module, internal_imports, external_imports)

        return internal_imports, external_imports

    def _categorize_import(self, import_name: str, internal: List[str], external: List[str]):
        """将import分类为内部或外部"""
        found = False
        for key in self.module_map:
            if import_name.startswith(key) or key.startswith(import_name):
                if self.module_map[key] in self.py_files:
                    internal.append(self.module_map[key])
                    found = True
                    break

        if not found:
            external.append(import_name.split('.')[0])

    def get_file_content(self, file_path: Path) -> str:
        """读取文件内容"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Warning: Failed to read {file_path}: {e}")
            return ""


class FileMerger:
    """合并Python文件并生成self-contained版本"""

    def __init__(self, repo_path: str, output_dir: str):
        self.repo_path = Path(repo_path)
        self.repo_name = self.repo_path.name
        self.analyzer = ImportAnalyzer(repo_path)
        self.output_dir = Path(output_dir) / self.repo_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def merge_file(self, module_name: str) -> Tuple[str, Set[str]]:
        """合并单个文件及其所有依赖（去重版本）

        Returns:
            (merged_content, external_dependencies)
        """
        visited = OrderedDict()  # 保持顺序，用于去重
        external_deps = set()
        processing = set()  # 用于检测循环依赖

        def process_module(mod_name: str, level: int = 0) -> int:
            """递归处理模块依赖

            Returns:
                实际的依赖深度级别
            """
            # 已处理过，直接返回（去重关键点）
            if mod_name in visited:
                return visited[mod_name][1]  # 返回已记录的level

            # 检测循环依赖
            if mod_name in processing:
                print(f"  Warning: Circular dependency detected: {mod_name}")
                return level

            if mod_name not in self.analyzer.py_files:
                return level

            processing.add(mod_name)

            file_path = self.analyzer.py_files[mod_name]
            internal_imports, external_imports = self.analyzer.parse_imports(file_path)

            # 去重内部依赖
            unique_internal_imports = list(dict.fromkeys(internal_imports))

            # 先处理依赖（深度优先），计算最大深度
            max_dep_level = level
            for dep in unique_internal_imports:
                if dep != mod_name:  # 避免自引用
                    dep_level = process_module(dep, level + 1)
                    max_dep_level = max(max_dep_level, dep_level)

            # 添加当前模块（只添加一次）
            content = self.analyzer.get_file_content(file_path)
            visited[mod_name] = (content, level, unique_internal_imports)
            external_deps.update(external_imports)

            processing.remove(mod_name)
            return level

        # 开始处理
        process_module(module_name)

        # 按依赖顺序重新排列（被依赖的排在前面）
        sorted_modules = self._topological_sort(visited)

        # 生成合并后的内容
        merged_content = self._generate_merged_content(sorted_modules, module_name)

        return merged_content, external_deps

    def _topological_sort(self, visited: OrderedDict) -> OrderedDict:
        """拓扑排序，确保依赖在前，依赖者在后"""
        # 建立依赖图
        graph = {mod: set(deps) & set(visited.keys())
                 for mod, (_, _, deps) in visited.items()}

        # 计算入度
        in_degree = {mod: 0 for mod in visited}
        for mod, deps in graph.items():
            for dep in deps:
                if dep in in_degree:
                    in_degree[mod] += 1

        # Kahn's algorithm
        sorted_modules = OrderedDict()
        queue = [mod for mod, degree in in_degree.items() if degree == 0]

        while queue:
            # 优先处理入度为0的节点
            current = queue.pop(0)
            content, level, deps = visited[current]
            sorted_modules[current] = (content, level, deps)

            # 更新入度
            for mod, mod_deps in graph.items():
                if current in mod_deps:
                    in_degree[mod] -= 1
                    if in_degree[mod] == 0 and mod not in sorted_modules:
                        queue.append(mod)

        # 处理可能的循环依赖遗留
        for mod in visited:
            if mod not in sorted_modules:
                content, level, deps = visited[mod]
                sorted_modules[mod] = (content, level, deps)

        return sorted_modules

    def _generate_merged_content(self, visited: OrderedDict, main_module: str) -> str:
        """生成带层级注释的合并内容"""
        lines = []
        lines.append("# " + "=" * 78)
        lines.append(f"# Self-contained merged file for: {main_module}")
        lines.append(f"# Repository: {self.repo_name}")
        lines.append(f"# Generated by repository file merger")
        lines.append(f"# Total modules included: {len(visited)}")
        lines.append("# " + "=" * 78)
        lines.append("")
        lines.append("# Import dependency tree (deduplicated):")

        # 生成依赖树
        for mod_name, (content, level, deps) in visited.items():
            indent = "  " * level
            marker = "★" if mod_name == main_module else "└─"
            lines.append(f"# {indent}{marker} {mod_name}")
            if deps:
                valid_deps = [dep for dep in deps if dep in visited]
                for dep in valid_deps:
                    lines.append(f"#   {indent}  ├─ imports: {dep}")

        lines.append("# " + "=" * 78)
        lines.append("")

        # 添加所有模块内容
        for idx, (mod_name, (content, level, deps)) in enumerate(visited.items(), 1):
            lines.append("")
            lines.append("# " + "-" * 78)
            lines.append(f"# Module [{idx}/{len(visited)}]: {mod_name}")
            lines.append(f"# Dependencies: {', '.join(deps) if deps else 'None'}")
            lines.append("# " + "-" * 78)
            lines.append("")
            lines.append(content)
            lines.append("")

        return "\n".join(lines)

    def process_all_files(self) -> Dict[str, any]:
        """处理所有Python文件

        Returns:
            处理统计信息
        """
        stats = {
            'repo_name': self.repo_name,
            'total_files': len(self.analyzer.py_files),
            'processed': 0,
            'failed': 0,
            'all_external_deps': set()
        }

        print(f"\n{'=' * 80}")
        print(f"Processing repository: {self.repo_name}")
        print(f"Found {len(self.analyzer.py_files)} Python files")
        print(f"Output directory: {self.output_dir}")
        print("-" * 80)

        for module_name, file_path in self.analyzer.py_files.items():
            try:
                print(f"  Processing: {module_name}")

                # 合并文件
                merged_content, external_deps = self.merge_file(module_name)

                # 生成输出文件名
                safe_name = module_name.replace(".", "_")
                output_py = self.output_dir / f"{safe_name}_merged.py"
                output_txt = self.output_dir / f"{safe_name}_deps.txt"

                # 写入合并后的Python文件
                with open(output_py, 'w', encoding='utf-8') as f:
                    f.write(merged_content)

                # 写入外部依赖列表
                with open(output_txt, 'w', encoding='utf-8') as f:
                    f.write(f"External dependencies for: {module_name}\n")
                    f.write(f"Repository: {self.repo_name}\n")
                    f.write("=" * 60 + "\n\n")
                    if external_deps:
                        for dep in sorted(external_deps):
                            f.write(f"{dep}\n")
                    else:
                        f.write("No external dependencies found.\n")

                print(f"    ✓ Generated: {output_py.name}")
                print(f"    ✓ External deps: {len(external_deps)}")

                stats['processed'] += 1
                stats['all_external_deps'].update(external_deps)

            except Exception as e:
                print(f"    ✗ Failed: {e}")
                stats['failed'] += 1

        # 生成仓库级别的依赖汇总
        self._generate_repo_summary(stats)

        return stats

    def _generate_repo_summary(self, stats: Dict):
        """生成仓库级别的依赖汇总"""
        summary_file = self.output_dir / "_REPO_SUMMARY.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"Repository Summary: {self.repo_name}\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Total Python files: {stats['total_files']}\n")
            f.write(f"Successfully processed: {stats['processed']}\n")
            f.write(f"Failed: {stats['failed']}\n\n")
            f.write("All external dependencies:\n")
            f.write("-" * 40 + "\n")
            for dep in sorted(stats['all_external_deps']):
                f.write(f"  {dep}\n")
        print(f"  ✓ Generated repository summary: {summary_file.name}")


class MultiRepoProcessor:
    """处理多个仓库的批量处理器"""

    def __init__(self, parent_dir: str, output_dir: str):
        self.parent_dir = Path(parent_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def find_repos(self) -> List[Path]:
        """找到所有包含Python文件的子目录（视为仓库）"""
        repos = []
        for item in self.parent_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                # 检查是否包含Python文件
                py_files = list(item.rglob("*.py"))
                # 排除__pycache__
                py_files = [f for f in py_files if "__pycache__" not in str(f)]
                if py_files:
                    repos.append(item)
        return repos

    def process_all_repos(self):
        """处理所有仓库"""
        repos = self.find_repos()

        print("=" * 80)
        print(f"Multi-Repository Processor")
        print(f"Parent directory: {self.parent_dir}")
        print(f"Found {len(repos)} repositories with Python files")
        print(f"Output directory: {self.output_dir}")
        print("=" * 80)

        all_stats = []

        for repo in repos:
            try:
                merger = FileMerger(str(repo), str(self.output_dir))
                stats = merger.process_all_files()
                all_stats.append(stats)
            except Exception as e:
                print(f"Error processing repository {repo.name}: {e}")
                all_stats.append({
                    'repo_name': repo.name,
                    'error': str(e)
                })

        # 生成总体汇总
        self._generate_global_summary(all_stats)

        print("\n" + "=" * 80)
        print("✓ All repositories processed!")
        print(f"✓ Output directory: {self.output_dir.absolute()}")
        print("=" * 80)

    def _generate_global_summary(self, all_stats: List[Dict]):
        """生成所有仓库的汇总报告"""
        summary_file = self.output_dir / "_GLOBAL_SUMMARY.txt"

        total_files = 0
        total_processed = 0
        total_failed = 0
        all_deps = set()

        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("Global Summary - All Repositories\n")
            f.write("=" * 60 + "\n\n")

            f.write("Repository Details:\n")
            f.write("-" * 40 + "\n")

            for stats in all_stats:
                if 'error' in stats:
                    f.write(f"\n{stats['repo_name']}: ERROR - {stats['error']}\n")
                else:
                    f.write(f"\n{stats['repo_name']}:\n")
                    f.write(f"  - Total files: {stats['total_files']}\n")
                    f.write(f"  - Processed: {stats['processed']}\n")
                    f.write(f"  - Failed: {stats['failed']}\n")
                    f.write(f"  - External deps: {len(stats['all_external_deps'])}\n")

                    total_files += stats['total_files']
                    total_processed += stats['processed']
                    total_failed += stats['failed']
                    all_deps.update(stats['all_external_deps'])

            f.write("\n" + "=" * 60 + "\n")
            f.write("Overall Statistics:\n")
            f.write(f"  - Total repositories: {len(all_stats)}\n")
            f.write(f"  - Total Python files: {total_files}\n")
            f.write(f"  - Successfully processed: {total_processed}\n")
            f.write(f"  - Failed: {total_failed}\n")
            f.write(f"  - Unique external dependencies: {len(all_deps)}\n")

            f.write("\n" + "=" * 60 + "\n")
            f.write("All External Dependencies (across all repos):\n")
            f.write("-" * 40 + "\n")
            for dep in sorted(all_deps):
                f.write(f"  {dep}\n")

        print(f"\n✓ Generated global summary: {summary_file.name}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Merge Python files in repositories into self-contained versions"
    )
    parser.add_argument(
        "path",
        help="Path to a repository or parent directory containing multiple repositories"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="./merged_output",
        help="Output directory (default: ./merged_output)"
    )
    parser.add_argument(
        "--multi",
        "-m",
        action="store_true",
        help="Process multiple repositories in subdirectories"
    )

    args = parser.parse_args()

    # 验证输入路径
    if not os.path.exists(args.path):
        print(f"Error: Path does not exist: {args.path}")
        sys.exit(1)

    if args.multi:
        # 批量处理多个仓库
        processor = MultiRepoProcessor(args.path, args.output)
        processor.process_all_repos()
    else:
        # 单仓库处理
        merger = FileMerger(args.path, args.output)
        merger.process_all_files()

    print("\n✓ All files processed successfully!")
    print(f"✓ Output directory: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
    # python step_1_merge_repo_1220.py /data/yubo/datasets/collected_sc_1214 -m -o /data/yubo/datasets/output_sc_1214
