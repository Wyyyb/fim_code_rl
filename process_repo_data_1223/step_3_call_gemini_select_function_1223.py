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
import time
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
"""


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
            print_response: bool = True,
            wait_seconds: float = 2.0,
            min_line_num: int = 50,
            max_line_num: int = 1000,
            min_func_num: int = 3
    ):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.gemini_client = GeminiClient(model=model)
        self.function_extractor = FunctionExtractor()
        self.print_response = print_response
        self.wait_seconds = wait_seconds

        # Filter parameters
        self.min_line_num = min_line_num
        self.max_line_num = max_line_num
        self.min_func_num = min_func_num

        # Global function ID counter
        self.function_id = 0

        # Load checkpoint if exists
        self.processed_samples = set()
        self.skipped_samples = set()  # Track skipped samples separately
        self.results = []  # Stores sample-level results
        self._load_checkpoint()

    def _load_checkpoint(self):
        """Load checkpoint data if exists."""
        if self.checkpoint_path.exists():
            try:
                with open(self.checkpoint_path, 'r', encoding='utf-8') as f:
                    checkpoint = json.load(f)
                self.processed_samples = set(checkpoint.get('processed_samples', []))
                self.skipped_samples = set(checkpoint.get('skipped_samples', []))
                self.results = checkpoint.get('results', [])
                self.function_id = checkpoint.get('next_function_id', 0)

                # Count total functions extracted
                total_funcs = sum(
                    len(r.get('selected_function_list', [])) for r in self.results
                )
                logger.info(f"Loaded checkpoint:")
                logger.info(f"  - Processed samples: {len(self.processed_samples)}")
                logger.info(f"  - Skipped samples: {len(self.skipped_samples)}")
                logger.info(f"  - Results saved: {len(self.results)}")
                logger.info(f"  - Functions extracted: {total_funcs}")
                logger.info(f"  - Next function ID: {self.function_id}")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")

    def _save_checkpoint(self):
        """Save current progress to checkpoint file."""
        checkpoint = {
            'processed_samples': list(self.processed_samples),
            'skipped_samples': list(self.skipped_samples),
            'results': self.results,
            'next_function_id': self.function_id
        }

        # Write to temp file first, then rename (atomic operation)
        temp_path = self.checkpoint_path.with_suffix('.tmp')
        try:
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(checkpoint, f, ensure_ascii=False, indent=2)
            # Atomic rename
            temp_path.replace(self.checkpoint_path)
            logger.debug(f"Checkpoint saved: {len(self.processed_samples)} processed, {len(self.results)} results")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            if temp_path.exists():
                temp_path.unlink()

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

    def _print_gemini_response(self, sample_id, response: str):
        """Print Gemini response content."""
        print("\n" + "=" * 80)
        print(f"📝 Gemini Response for Sample: {sample_id}")
        print("=" * 80)
        print(response)
        print("=" * 80 + "\n")

    def _check_sample_eligibility(self, sample: dict) -> tuple[bool, str]:
        """
        Check if a sample meets the filtering criteria.

        Returns:
            tuple: (is_eligible, reason)
        """
        line_num = sample.get('line_num', 0)
        func_num = sample.get('func_num', 0)

        if line_num < self.min_line_num:
            return False, f"line_num ({line_num}) < min_line_num ({self.min_line_num})"

        if line_num > self.max_line_num:
            return False, f"line_num ({line_num}) > max_line_num ({self.max_line_num})"

        if func_num < self.min_func_num:
            return False, f"func_num ({func_num}) < min_func_num ({self.min_func_num})"

        return True, "passed"

    def process_single_sample(self, sample: dict) -> Optional[dict]:
        """
        Process a single code sample and return the enriched sample with selected functions.

        Returns:
            The original sample dict with added fields, or None if skipped/failed.
        """
        sample_id = sample['sample_id']
        code_content = sample['code_content']

        # Build prompt
        prompt = PROMPT_TEMPLATE + code_content

        # Call Gemini API
        try:
            response = self.gemini_client.get_response(prompt)

            # Print Gemini response if enabled
            if self.print_response:
                self._print_gemini_response(sample_id, response)

            # Wait specified seconds
            if self.wait_seconds > 0:
                logger.info(f"⏳ Waiting {self.wait_seconds} seconds before next request...")
                time.sleep(self.wait_seconds)

        except Exception as e:
            logger.error(f"Failed to get Gemini response for sample {sample_id}: {e}")
            return None

        # Parse response
        parsed_response = self._parse_gemini_response(response)
        if not parsed_response:
            logger.warning(f"Failed to parse response for sample {sample_id}")
            return None

        # Check if code is suitable
        code_eval = parsed_response.get('code_evaluation', {})

        # Create result based on original sample
        result_sample = sample.copy()
        result_sample['gemini_full_response'] = response
        result_sample['code_evaluation'] = code_eval
        result_sample['selected_function_list'] = []

        if not code_eval.get('is_suitable', False):
            logger.info(f"Sample {sample_id} not suitable: {code_eval.get('rejection_reason', 'Unknown')}")
            return result_sample

        # Process selected functions
        selected_functions = parsed_response.get('selected_functions', [])

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

            # Create function item
            function_item = {
                'function_id': self.function_id,
                'function_name': function_name,
                'function_code': func_data['code'],
                'masked_code': masked_code,
                'start_line': func_data['start_line'],
                'end_line': func_data['end_line'],
                'difficulty_score': func_info.get('difficulty_score'),
                'selection_reason': func_info.get('reason')
            }

            result_sample['selected_function_list'].append(function_item)
            self.function_id += 1

        return result_sample

    def run(self):
        """Run the FIM data generation pipeline."""
        # Load input data
        logger.info(f"Loading input data from {self.input_path}")
        with open(self.input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        total_samples = len(data)

        # Get all sample IDs that have been handled (processed or skipped)
        handled_samples = self.processed_samples | self.skipped_samples

        # Filter out already handled samples
        remaining_samples = [s for s in data if s['sample_id'] not in handled_samples]

        logger.info(f"Total samples in file: {total_samples}")
        logger.info(f"Already processed (sent to Gemini): {len(self.processed_samples)}")
        logger.info(f"Already skipped (filter criteria): {len(self.skipped_samples)}")
        logger.info(f"Remaining to handle: {len(remaining_samples)}")
        logger.info(
            f"Filter criteria: line_num in [{self.min_line_num}, {self.max_line_num}], func_num >= {self.min_func_num}")

        # Group by repo for progress tracking
        repo_samples = {}
        for sample in remaining_samples:
            repo_id = sample['repo_id']
            if repo_id not in repo_samples:
                repo_samples[repo_id] = []
            repo_samples[repo_id].append(sample)

        # Statistics for this run
        skipped_this_run = 0
        processed_this_run = 0

        # Calculate total functions extracted so far
        total_functions_extracted = sum(
            len(r.get('selected_function_list', [])) for r in self.results
        )

        # Process samples
        total_repos = len(repo_samples)
        with tqdm(total=len(remaining_samples), desc="Processing samples") as pbar:
            for repo_idx, (repo_id, samples) in enumerate(repo_samples.items()):
                logger.info(f"\n{'=' * 60}")
                logger.info(f"Processing Repo {repo_idx + 1}/{total_repos} (repo_id: {repo_id})")
                logger.info(f"Samples in this repo: {len(samples)}")

                for sample_idx, sample in enumerate(samples):
                    sample_id = sample['sample_id']

                    # Check eligibility before calling Gemini
                    is_eligible, reason = self._check_sample_eligibility(sample)

                    if not is_eligible:
                        logger.info(f"  Sample {sample_id} skipped: {reason}")
                        skipped_this_run += 1
                        self.skipped_samples.add(sample_id)

                        # Save checkpoint after each skipped sample
                        self._save_checkpoint()

                        pbar.update(1)
                        continue

                    # Update progress description
                    pbar.set_description(
                        f"Repo {repo_idx + 1}/{total_repos} | "
                        f"Sample {sample_idx + 1}/{len(samples)} | "
                        f"Funcs: {total_functions_extracted}"
                    )

                    # Process sample
                    try:
                        result = self.process_single_sample(sample)
                        if result:
                            self.results.append(result)
                            processed_this_run += 1
                            num_funcs = len(result.get('selected_function_list', []))
                            total_functions_extracted += num_funcs

                            if num_funcs > 0:
                                logger.info(f"  ✅ Sample {sample_id}: extracted {num_funcs} functions")
                            else:
                                logger.info(f"  ⚪ Sample {sample_id}: no functions selected")

                            # Mark as processed
                            self.processed_samples.add(sample_id)
                        else:
                            # API call failed or parse failed, still mark to avoid retry
                            logger.warning(f"  ❌ Sample {sample_id}: processing failed")
                            self.processed_samples.add(sample_id)

                    except Exception as e:
                        logger.error(f"Error processing sample {sample_id}: {e}")
                        # Mark as processed to avoid infinite retry
                        self.processed_samples.add(sample_id)

                    # Save checkpoint after each processed sample
                    self._save_checkpoint()

                    pbar.update(1)

        # Save final output
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Saving {len(self.results)} processed samples to {self.output_path}")
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)

        # Print summary
        self._print_summary(skipped_this_run, processed_this_run, total_functions_extracted)

    def _print_summary(self, skipped_this_run: int, processed_this_run: int, total_functions: int):
        """Print processing summary."""
        print("\n" + "=" * 60)
        print("Processing Summary")
        print("=" * 60)
        print(f"\n📊 Overall Statistics:")
        print(f"  Total samples processed (sent to Gemini): {len(self.processed_samples)}")
        print(f"  Total samples skipped (filter criteria): {len(self.skipped_samples)}")
        print(f"  Total results saved: {len(self.results)}")
        print(f"  Total functions extracted: {total_functions}")

        print(f"\n📈 This Run:")
        print(f"  Samples processed: {processed_this_run}")
        print(f"  Samples skipped: {skipped_this_run}")

        if self.results:
            # Count samples with functions
            samples_with_funcs = sum(1 for r in self.results if r.get('selected_function_list'))
            print(f"\n📁 Results Breakdown:")
            print(f"  Samples with selected functions: {samples_with_funcs}")
            print(f"  Samples without functions: {len(self.results) - samples_with_funcs}")

            # Statistics by difficulty
            difficulty_counts = {}
            for result in self.results:
                for func in result.get('selected_function_list', []):
                    score = func.get('difficulty_score', 'N/A')
                    difficulty_counts[score] = difficulty_counts.get(score, 0) + 1

            if difficulty_counts:
                print("\n📊 Difficulty Distribution:")
                for score in sorted(difficulty_counts.keys(), key=lambda x: (isinstance(x, str), x)):
                    print(f"    Score {score}: {difficulty_counts[score]} functions")

            # Average functions per sample
            if samples_with_funcs > 0:
                avg_funcs = total_functions / samples_with_funcs
                print(f"\n📈 Average functions per suitable sample: {avg_funcs:.2f}")

            # Code evaluation statistics
            suitable_count = sum(
                1 for r in self.results
                if r.get('code_evaluation', {}).get('is_suitable', False)
            )
            print(f"\n✅ Samples deemed suitable by Gemini: {suitable_count}")
            print(f"❌ Samples deemed not suitable: {len(self.results) - suitable_count}")

        print("\n" + "=" * 60)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate FIM training data from Python code")
    parser.add_argument(
        "--input", "-i",
        default="/data/yubo/datasets/extracted_python_files_1223.json",
        help="Path to input JSON file"
    )
    parser.add_argument(
        "--output", "-o",
        default="/data/yubo/datasets/fim_training_data.json",
        help="Path to output JSON file"
    )
    parser.add_argument(
        "--checkpoint", "-c",
        default="/data/yubo/datasets/fim_checkpoint.json",
        help="Path to checkpoint file"
    )
    parser.add_argument(
        "--model", "-m",
        default="gemini-3-flash-preview",
        help="Gemini model to use"
    )
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
    parser.add_argument(
        "--min-lines",
        type=int,
        default=50,
        help="Minimum line_num to process a sample (default: 50)"
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=1000,
        help="Maximum line_num to process a sample (default: 1000)"
    )
    parser.add_argument(
        "--min-funcs",
        type=int,
        default=3,
        help="Minimum func_num to process a sample (default: 3)"
    )

    args = parser.parse_args()

    generator = FIMDataGenerator(
        input_path=args.input,
        output_path=args.output,
        checkpoint_path=args.checkpoint,
        model=args.model,
        print_response=args.print_response,
        wait_seconds=args.wait,
        min_line_num=args.min_lines,
        max_line_num=args.max_lines,
        min_func_num=args.min_funcs
    )

    generator.run()


if __name__ == "__main__":
    main()