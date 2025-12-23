#!/usr/bin/env python3
"""
FIM (Fill-in-the-Middle) Training Data Generator

This script processes Python code files and uses Gemini API to select appropriate
functions for FIM training tasks.
"""

import os
import json
import logging
import re
import ast
import time  # 新增：用于等待
from pathlib import Path
from typing import Optional
from tqdm import tqdm
from google import genai

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# English prompt template
PROMPT_TEMPLATE = """You are a code analysis expert. Your task is to help construct a Fill-in-the-Middle (FIM) training dataset.

## Task Background

Fill-in-the-Middle is a code completion training task. Our goal is to select appropriate functions from a complete Python file to "mask out," and then train a model to complete the masked function body based on the surrounding context (prefix code + suffix code). Ideally, the selected function should:
- Have calling relationships with other code in the file (e.g., if function A calls function B, and function B calls function C, then masking function B is a good choice because the context provides sufficient information to infer B's functionality)
- Have moderate completion difficulty—neither too trivial (like simple getters/setters) nor impossible to reasonably infer due to insufficient context

## Your Task

Please analyze the following Python code and complete two steps:

### Step 1: Code Quality Evaluation

Evaluate whether this code is suitable for the FIM task based on the following three dimensions. Provide a score from 1-5 for each (5 being the highest):

1. **Complexity**: Does the code have sufficient logical complexity? Overly simple code (e.g., containing only a few independent utility functions) is not suitable.
2. **Quality**: Is the code syntactically correct, functionally complete, and well-styled? Code with obvious errors or incomplete implementations is not suitable.
3. **Cohesion**: Do the functions in the code have calling relationships or logical connections with each other? If all functions are completely independent and unrelated utility functions, it is not suitable for this task.

Based on the scores from all three dimensions, provide a final verdict: **Suitable** or **Not Suitable** for the FIM task.
- Suggested criteria: If any dimension scores below 2, or if the average score is below 3, the verdict should be "Not Suitable"

### Step 2: Select Functions to Mask

If Step 1 verdict is "Suitable," please select functions that are appropriate as FIM completion targets. Selection criteria:

1. **Length requirement**: The function body should be approximately 10-100 lines (too short lacks training value; too long increases completion difficulty)
2. **Context sufficiency**: After masking the function, the remaining code (including call sites, functions called by the target, related comments, type hints, etc.) should provide enough information to infer the function's purpose and implementation
3. **Moderate difficulty**: Should not be simple boilerplate code (like simple assignments or direct returns), nor should it require domain expertise or external information to implement

For each selected function, provide:
- Function name
- Difficulty score (1-5, where 3 is moderate)
- Reason for selection (briefly explain why this function is suitable as a FIM task target)

## Output Format

Please output strictly in the following JSON format:

{
  "code_evaluation": {
    "complexity_score": <1-5>,
    "complexity_reason": "<brief explanation>",
    "quality_score": <1-5>,
    "quality_reason": "<brief explanation>",
    "cohesion_score": <1-5>,
    "cohesion_reason": "<brief explanation>",
    "average_score": <calculated average>,
    "is_suitable": <true/false>,
    "rejection_reason": "<if not suitable, state the main reason; if suitable, set to null>"
  },
  "selected_functions": [
    {
      "function_name": "<function name>",
      "difficulty_score": <1-5>,
      "reason": "<reason for selecting this function>"
    }
  ]
}

Notes:
- If is_suitable is false, then selected_functions should be an empty array []
- Multiple functions can be selected; each will generate an independent training sample
- If the code is suitable but no functions meet the criteria, selected_functions can also be an empty array

## Code to Analyze

```python
{code_content}
```"""


class GeminiClient:
    """Gemini API client for code analysis."""

    def __init__(self, model: str = "gemini-2.5-flash-preview-05-20"):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        self.client = genai.Client()
        self.model = model

    def get_response(self, prompt: str, temperature: float = 0.3) -> str:
        """Get response from Gemini API."""
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config={
                    "temperature": temperature,
                    "top_p": 0.9,
                },
            )
            return response.text
        except Exception as e:
            logger.error(f"Gemini API call failed: {str(e)}")
            raise


class FunctionExtractor:
    """Extract and manipulate functions from Python code."""

    @staticmethod
    def extract_function_info(code: str, function_name: str) -> Optional[dict]:
        """
        Extract function code and its position from the source code.

        Returns:
            dict with 'code', 'start_line', 'end_line', 'start_col', 'end_col'
            or None if function not found
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            # Fallback to regex-based extraction for code with syntax errors
            return FunctionExtractor._extract_function_regex(code, function_name)

        lines = code.split('\n')

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                start_line = node.lineno - 1  # 0-indexed
                end_line = node.end_lineno  # 1-indexed, so this is correct for slicing

                # Extract the function code
                func_lines = lines[start_line:end_line]
                func_code = '\n'.join(func_lines)

                return {
                    'code': func_code,
                    'start_line': start_line,
                    'end_line': end_line,
                    'function_name': function_name
                }

        # Try regex as fallback
        return FunctionExtractor._extract_function_regex(code, function_name)

    @staticmethod
    def _extract_function_regex(code: str, function_name: str) -> Optional[dict]:
        """Regex-based function extraction as fallback."""
        lines = code.split('\n')

        # Pattern to match function definition
        func_pattern = re.compile(rf'^(\s*)def\s+{re.escape(function_name)}\s*\(')

        start_line = None
        base_indent = None

        for i, line in enumerate(lines):
            match = func_pattern.match(line)
            if match:
                start_line = i
                base_indent = len(match.group(1))
                break

        if start_line is None:
            return None

        # Find the end of the function
        end_line = start_line + 1
        for i in range(start_line + 1, len(lines)):
            line = lines[i]

            # Skip empty lines
            if not line.strip():
                end_line = i + 1
                continue

            # Check indentation
            current_indent = len(line) - len(line.lstrip())

            # If we hit a line with same or less indentation (and it's not empty),
            # the function has ended
            if current_indent <= base_indent and line.strip():
                break

            end_line = i + 1

        func_lines = lines[start_line:end_line]
        func_code = '\n'.join(func_lines)

        return {
            'code': func_code,
            'start_line': start_line,
            'end_line': end_line,
            'function_name': function_name
        }

    @staticmethod
    def mask_function(code: str, function_name: str) -> Optional[str]:
        """
        Replace function body with a placeholder, keeping the signature.

        Returns the masked code or None if function not found.
        """
        func_info = FunctionExtractor.extract_function_info(code, function_name)
        if not func_info:
            return None

        lines = code.split('\n')
        func_lines = func_info['code'].split('\n')

        # Find the function signature (first line with 'def')
        signature_line = None
        for i, line in enumerate(func_lines):
            if 'def ' in line and function_name in line:
                signature_line = i
                break

        if signature_line is None:
            return None

        # Find where the signature ends (could be multi-line)
        signature_end = signature_line
        paren_count = 0
        for i in range(signature_line, len(func_lines)):
            line = func_lines[i]
            paren_count += line.count('(') - line.count(')')
            if paren_count <= 0 and ':' in line:
                signature_end = i
                break

        # Get the indentation of the function body
        body_indent = None
        for i in range(signature_end + 1, len(func_lines)):
            line = func_lines[i]
            if line.strip():  # Non-empty line
                body_indent = len(line) - len(line.lstrip())
                break

        if body_indent is None:
            body_indent = 4  # Default indentation

        # Construct the masked function
        signature_lines = func_lines[signature_line:signature_end + 1]
        masked_func = '\n'.join(signature_lines) + '\n' + ' ' * body_indent + '# <MASKED_FUNCTION_BODY>'

        # Replace in original code
        start_line = func_info['start_line']
        end_line = func_info['end_line']

        new_lines = lines[:start_line] + [masked_func] + lines[end_line:]

        return '\n'.join(new_lines)


class FIMDataGenerator:
    """Generate FIM training data from Python code files."""

    def __init__(
            self,
            input_path: str,
            output_path: str,
            checkpoint_path: str,
            model: str = "gemini-3-flash-preview",
            print_response: bool = True,  # 新增：是否打印响应
            wait_seconds: float = 2.0  # 新增：等待秒数
    ):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.gemini_client = GeminiClient(model=model)
        self.function_extractor = FunctionExtractor()
        self.print_response = print_response  # 新增
        self.wait_seconds = wait_seconds  # 新增

        # Global function ID counter
        self.function_id = 0

        # Load checkpoint if exists
        self.processed_samples = set()
        self.results = []
        self._load_checkpoint()

    def _load_checkpoint(self):
        """Load checkpoint data if exists."""
        if self.checkpoint_path.exists():
            try:
                with open(self.checkpoint_path, 'r', encoding='utf-8') as f:
                    checkpoint = json.load(f)
                self.processed_samples = set(checkpoint.get('processed_samples', []))
                self.results = checkpoint.get('results', [])
                self.function_id = checkpoint.get('next_function_id', 0)
                logger.info(f"Loaded checkpoint: {len(self.processed_samples)} samples processed, "
                            f"{len(self.results)} functions extracted")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")

    def _save_checkpoint(self):
        """Save current progress to checkpoint file."""
        checkpoint = {
            'processed_samples': list(self.processed_samples),
            'results': self.results,
            'next_function_id': self.function_id
        }
        with open(self.checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)

    def _extract_json_object(self, text: str) -> Optional[str]:
        """Extract a complete JSON object by matching balanced braces."""
        start_idx = text.find('{')
        if start_idx == -1:
            return None

        brace_count = 0
        in_string = False
        escape_next = False

        for i, char in enumerate(text[start_idx:], start=start_idx):
            if escape_next:
                escape_next = False
                continue

            if char == '\\':
                escape_next = True
                continue

            if char == '"' and not escape_next:
                in_string = not in_string
                continue

            if in_string:
                continue

            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    return text[start_idx:i + 1]

        return None

    def _parse_gemini_response(self, response: str) -> Optional[dict]:
        """Parse Gemini's JSON response."""
        if not response or not response.strip():
            logger.warning("Empty response from Gemini")
            return None

        # Try to extract JSON from markdown code blocks first
        code_block_match = re.search(r'```(?:json)?\s*\n?([\s\S]*?)\n?```', response)
        if code_block_match:
            json_str = code_block_match.group(1).strip()
        else:
            # Try to find JSON object by matching balanced braces
            json_str = self._extract_json_object(response)

        if not json_str:
            logger.warning(f"No JSON found in response. Response preview: {response[:300]}...")
            return None

        try:
            result = json.loads(json_str)
            if 'code_evaluation' not in result:
                logger.warning("Missing 'code_evaluation' in parsed response")
                return None
            return result
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            logger.error(f"Problematic JSON (first 500 chars): {json_str[:500]}")
            return None

    def _print_gemini_response(self, sample_id: str, response: str):
        """打印 Gemini 的响应内容"""
        print("\n" + "=" * 80)
        print(f"📝 Gemini Response for Sample: {sample_id}")
        print("=" * 80)
        print(response)
        print("=" * 80 + "\n")

    def process_single_sample(self, sample: dict) -> list[dict]:
        """
        Process a single code sample and return list of FIM training items.
        """
        sample_id = sample['sample_id']
        repo_id = sample['repo_id']
        code_content = sample['code_content']

        # Build prompt
        prompt = PROMPT_TEMPLATE.format(code_content=code_content)

        # Call Gemini API
        try:
            response = self.gemini_client.get_response(prompt)

            # 新增：打印 Gemini 响应
            if self.print_response:
                self._print_gemini_response(sample_id, response)

            # 新增：等待指定秒数
            if self.wait_seconds > 0:
                logger.info(f"⏳ Waiting {self.wait_seconds} seconds before next request...")
                time.sleep(self.wait_seconds)

        except Exception as e:
            logger.error(f"Failed to get Gemini response for sample {sample_id}: {e}")
            return []

        # Parse response
        parsed_response = self._parse_gemini_response(response)
        if not parsed_response:
            logger.warning(f"Failed to parse response for sample {sample_id}")
            return []

        # Check if code is suitable
        code_eval = parsed_response.get('code_evaluation', {})
        if not code_eval.get('is_suitable', False):
            logger.info(f"Sample {sample_id} not suitable: {code_eval.get('rejection_reason', 'Unknown')}")
            return []

        # Process selected functions
        selected_functions = parsed_response.get('selected_functions', [])
        results = []

        for func_info in selected_functions:
            function_name = func_info.get('function_name')
            if not function_name:
                continue

            # Extract function code
            func_data = self.function_extractor.extract_function_info(code_content, function_name)
            if not func_data:
                logger.warning(f"Could not extract function '{function_name}' from sample {sample_id}")
                continue

            # Create masked version
            masked_code = self.function_extractor.mask_function(code_content, function_name)
            if not masked_code:
                logger.warning(f"Could not mask function '{function_name}' in sample {sample_id}")
                continue

            # Create result item
            result_item = {
                'repo_id': repo_id,
                'sample_id': sample_id,
                'function_id': self.function_id,
                'function_name': function_name,
                'function_code': func_data['code'],
                'masked_code': masked_code,
                'difficulty_score': func_info.get('difficulty_score'),
                'selection_reason': func_info.get('reason'),
                'gemini_full_response': response,
                'code_evaluation': code_eval
            }

            results.append(result_item)
            self.function_id += 1

        return results

    def run(self):
        """Run the FIM data generation pipeline."""
        # Load input data
        logger.info(f"Loading input data from {self.input_path}")
        with open(self.input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        total_samples = len(data)
        remaining_samples = [s for s in data if s['sample_id'] not in self.processed_samples]

        logger.info(f"Total samples: {total_samples}")
        logger.info(f"Already processed: {len(self.processed_samples)}")
        logger.info(f"Remaining: {len(remaining_samples)}")

        # Group by repo for progress tracking
        repo_samples = {}
        for sample in remaining_samples:
            repo_id = sample['repo_id']
            if repo_id not in repo_samples:
                repo_samples[repo_id] = []
            repo_samples[repo_id].append(sample)

        # Process samples
        total_repos = len(repo_samples)

        with tqdm(total=len(remaining_samples), desc="Processing samples") as pbar:
            for repo_idx, (repo_id, samples) in enumerate(repo_samples.items()):
                logger.info(f"\n{'=' * 60}")
                logger.info(f"Processing Repo {repo_idx + 1}/{total_repos} (repo_id: {repo_id})")
                logger.info(f"Samples in this repo: {len(samples)}")

                for sample_idx, sample in enumerate(samples):
                    sample_id = sample['sample_id']

                    # Update progress description
                    pbar.set_description(
                        f"Repo {repo_idx + 1}/{total_repos} | "
                        f"Sample {sample_idx + 1}/{len(samples)} | "
                        f"Functions: {len(self.results)}"
                    )

                    # Process sample
                    try:
                        new_items = self.process_single_sample(sample)
                        self.results.extend(new_items)

                        if new_items:
                            logger.info(f"  Sample {sample_id}: extracted {len(new_items)} functions")

                    except Exception as e:
                        logger.error(f"Error processing sample {sample_id}: {e}")

                    # Mark as processed
                    self.processed_samples.add(sample_id)

                    # Save checkpoint periodically
                    if len(self.processed_samples) % 10 == 0:
                        self._save_checkpoint()

                    pbar.update(1)

        # Final save
        self._save_checkpoint()

        # Save final results
        logger.info(f"\nSaving {len(self.results)} FIM training items to {self.output_path}")
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)

        # Print summary
        self._print_summary()

    def _print_summary(self):
        """Print processing summary."""
        print("\n" + "=" * 60)
        print("Processing Summary")
        print("=" * 60)
        print(f"Total samples processed: {len(self.processed_samples)}")
        print(f"Total functions extracted: {len(self.results)}")

        if self.results:
            # Statistics by difficulty
            difficulty_counts = {}
            for item in self.results:
                score = item.get('difficulty_score', 'N/A')
                difficulty_counts[score] = difficulty_counts.get(score, 0) + 1

            print("\nDifficulty Distribution:")
            for score in sorted(difficulty_counts.keys(), key=lambda x: (isinstance(x, str), x)):
                print(f"  Score {score}: {difficulty_counts[score]} functions")

            # Functions per sample
            sample_counts = {}
            for item in self.results:
                sid = item['sample_id']
                sample_counts[sid] = sample_counts.get(sid, 0) + 1

            avg_funcs = sum(sample_counts.values()) / len(sample_counts) if sample_counts else 0
            print(f"\nAverage functions per suitable sample: {avg_funcs:.2f}")

        print("=" * 60)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate FIM training data from Python code")
    parser.add_argument(
        "--input", "-i",
        default="/data/yubo/datasets/step_2_extracted_python_files_1223.json",
        help="Path to input JSON file"
    )
    parser.add_argument(
        "--output", "-o",
        default="/data/yubo/datasets/step_3_fim_training_data.json",
        help="Path to output JSON file"
    )
    parser.add_argument(
        "--checkpoint", "-c",
        default="/data/yubo/datasets/step_3_fim_checkpoint.json",
        help="Path to checkpoint file"
    )
    parser.add_argument(
        "--model", "-m",
        default="gemini-3-flash-preview",
        help="Gemini model to use"
    )
    # 新增：命令行参数控制打印和等待
    parser.add_argument(
        "--print-response", "-p",
        action="store_true",
        default=True,
        help="Print Gemini response for each sample (default: True)"
    )
    parser.add_argument(
        "--no-print-response",
        action="store_false",
        dest="print_response",
        help="Disable printing Gemini response"
    )
    parser.add_argument(
        "--wait", "-w",
        type=float,
        default=2.0,
        help="Seconds to wait after each API call (default: 2.0)"
    )

    args = parser.parse_args()

    generator = FIMDataGenerator(
        input_path=args.input,
        output_path=args.output,
        checkpoint_path=args.checkpoint,
        model=args.model,
        print_response=args.print_response,  # 新增
        wait_seconds=args.wait  # 新增
    )

    generator.run()


if __name__ == "__main__":
    main()