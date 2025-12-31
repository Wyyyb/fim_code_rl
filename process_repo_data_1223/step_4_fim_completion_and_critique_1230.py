#!/usr/bin/env python3
"""
FIM Completion and Critique Pipeline

This script processes the selected functions from step 3 and:
1. Calls Gemini to complete the masked function (with CoT reasoning)
2. Calls Gemini again to critique the completion against ground truth (with CoT)

Supports sharding for parallel processing and checkpoint-based resumption.
"""

import os
import json
import logging
import re
import time
import argparse
from pathlib import Path
from typing import Optional
from tqdm import tqdm
from google import genai

# Optional wandb/weave integration
try:
    import wandb
    import weave

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️ wandb/weave not installed. Run: pip install wandb weave")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Gemini Pricing (per 1M tokens in USD) - Paid Tier
PRICING = {
    'input': 0.50,
    'output': 3.00,
}

# =============================================================================
# Prompt Templates
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

CRITIQUE_PROMPT = '''You are an expert code reviewer. Your task is to evaluate a function completion against the ground truth implementation.

## Task

A model was asked to complete a masked function based on surrounding code context. You need to:
1. Compare the completion with the ground truth
2. Analyze the differences and similarities
3. Score the completion across multiple dimensions

## Context (Code with Masked Function)

```python
{masked_code}
```

## Function Being Evaluated

Function name: `{function_name}`

## Ground Truth Implementation

```python
{ground_truth}
```

## Model's Completion

```python
{completion}
```

## Evaluation Criteria

Please evaluate the completion on these dimensions (1-5 scale, where 5 is best):

1. **Correctness (1-5)**: Does the implementation produce the same results as the ground truth for all inputs?
   - 5: Functionally identical
   - 4: Correct for most cases, minor edge case differences
   - 3: Partially correct, some important cases wrong
   - 2: Major functional differences
   - 1: Completely incorrect

2. **Executability (1-5)**: Can the code actually run without errors? Check for syntax errors, indentation issues, undefined variables, missing imports, type errors, etc.
   - 5: Code is fully executable, no errors
   - 4: Minor issues that are easy to fix (e.g., a small typo, slightly off indentation)
   - 3: Some issues but the main logic is executable (e.g., missing an edge case handler that would crash)
   - 2: Significant issues that would prevent execution (e.g., syntax errors, undefined variables)
   - 1: Code is pseudo-code or completely non-executable (e.g., placeholder comments, wrong language syntax)

3. **API Usage (1-5)**: Does it correctly use other functions, methods, and external APIs present in the codebase?
   - 5: All calls match ground truth
   - 4: Minor differences in API usage
   - 3: Some incorrect or missing API calls
   - 2: Major API usage errors
   - 1: Completely wrong API usage

4. **Readability (1-5)**: Is the code clean, well-formatted, and follows Python conventions?
   - 5: Excellent readability, possibly better than ground truth
   - 4: Good readability, comparable to ground truth
   - 3: Acceptable readability
   - 2: Poor readability
   - 1: Very hard to read

5. **Completeness (1-5)**: Does the implementation handle all cases the ground truth handles?
   - 5: Handles all cases including edge cases
   - 4: Handles main cases, may miss some edge cases
   - 3: Handles basic cases only
   - 2: Missing important cases
   - 1: Very incomplete

## Output Format

Please provide your analysis in the following JSON format:

```json
{{
  "reasoning": "<Your detailed step-by-step analysis comparing the two implementations>",
  "scores": {{
    "correctness": <1-5>,
    "correctness_reason": "<brief explanation>",
    "executability": <1-5>,
    "executability_reason": "<brief explanation, list specific issues if any>",
    "api_usage": <1-5>,
    "api_usage_reason": "<brief explanation>",
    "readability": <1-5>,
    "readability_reason": "<brief explanation>",
    "completeness": <1-5>,
    "completeness_reason": "<brief explanation>"
  }},
  "overall_score": <1-5>,
  "overall_reason": "<summary of the overall quality>",
  "key_differences": ["<list of main differences>"],
  "key_similarities": ["<list of main similarities>"]
}}
```

## Scoring Guidelines for Overall Score
- **5**: The completion is functionally equivalent to the ground truth. It would work correctly as a drop-in replacement.
- **4**: The completion is mostly correct with minor issues that wouldn't significantly impact functionality.
- **3**: The completion captures the main idea but has notable differences or issues.
- **2**: The completion has major problems but shows some understanding of the task.
- **1**: The completion is fundamentally incorrect or completely misses the point.

Note: A high correctness score but low executability score indicates the model understood the logic but made implementation mistakes (e.g., pseudo-code or syntax errors). This distinction is important for training data quality assessment.

Please provide your detailed analysis and scores.
'''


# =============================================================================
# Utility Classes
# =============================================================================

class TokenCounter:
    """Track token usage and calculate costs."""

    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.request_count = 0
        # Separate tracking for FIM and Critique
        self.fim_input_tokens = 0
        self.fim_output_tokens = 0
        self.fim_requests = 0
        self.critique_input_tokens = 0
        self.critique_output_tokens = 0
        self.critique_requests = 0

    def add_usage(self, input_tokens: int, output_tokens: int, call_type: str = "fim"):
        """Add token usage from a request."""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.request_count += 1

        if call_type == "fim":
            self.fim_input_tokens += input_tokens
            self.fim_output_tokens += output_tokens
            self.fim_requests += 1
        elif call_type == "critique":
            self.critique_input_tokens += input_tokens
            self.critique_output_tokens += output_tokens
            self.critique_requests += 1

    def get_cost(self) -> dict:
        """Calculate current costs."""
        input_cost = (self.total_input_tokens / 1_000_000) * PRICING['input']
        output_cost = (self.total_output_tokens / 1_000_000) * PRICING['output']
        total_cost = input_cost + output_cost

        return {
            'input_tokens': self.total_input_tokens,
            'output_tokens': self.total_output_tokens,
            'total_tokens': self.total_input_tokens + self.total_output_tokens,
            'input_cost': input_cost,
            'output_cost': output_cost,
            'total_cost': total_cost,
            'request_count': self.request_count,
            'fim_requests': self.fim_requests,
            'critique_requests': self.critique_requests,
        }

    def print_cost_summary(self):
        """Print a formatted cost summary."""
        cost = self.get_cost()
        print(f"\n💰 Token Usage & Cost:")
        print(f"   Input tokens:  {cost['input_tokens']:,} (${cost['input_cost']:.4f})")
        print(f"   Output tokens: {cost['output_tokens']:,} (${cost['output_cost']:.4f})")
        print(f"   Total tokens:  {cost['total_tokens']:,}")
        print(f"   FIM requests:  {cost['fim_requests']}")
        print(f"   Critique requests: {cost['critique_requests']}")
        print(f"   💵 Total cost: ${cost['total_cost']:.4f}")

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'total_input_tokens': self.total_input_tokens,
            'total_output_tokens': self.total_output_tokens,
            'request_count': self.request_count,
            'fim_input_tokens': self.fim_input_tokens,
            'fim_output_tokens': self.fim_output_tokens,
            'fim_requests': self.fim_requests,
            'critique_input_tokens': self.critique_input_tokens,
            'critique_output_tokens': self.critique_output_tokens,
            'critique_requests': self.critique_requests,
        }

    def load_from_dict(self, data: dict):
        """Load from dictionary."""
        self.total_input_tokens = data.get('total_input_tokens', 0)
        self.total_output_tokens = data.get('total_output_tokens', 0)
        self.request_count = data.get('request_count', 0)
        self.fim_input_tokens = data.get('fim_input_tokens', 0)
        self.fim_output_tokens = data.get('fim_output_tokens', 0)
        self.fim_requests = data.get('fim_requests', 0)
        self.critique_input_tokens = data.get('critique_input_tokens', 0)
        self.critique_output_tokens = data.get('critique_output_tokens', 0)
        self.critique_requests = data.get('critique_requests', 0)


class GeminiClient:
    """Gemini API client for code completion and critique."""

    def __init__(
            self,
            model: str = "gemini-3-flash-preview",
            token_counter: TokenCounter = None,
            use_wandb: bool = False
    ):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        self.client = genai.Client()
        self.model = model
        self.token_counter = token_counter
        self.use_wandb = use_wandb and WANDB_AVAILABLE

    def get_response(
            self,
            prompt: str,
            sample_id: str = None,
            call_type: str = "fim",
            temperature: float = 0.7
    ) -> tuple[str, dict]:
        """
        Get response from Gemini API.

        Args:
            prompt: The prompt to send
            sample_id: Identifier for logging
            call_type: "fim" or "critique"
            temperature: Sampling temperature

        Returns:
            tuple: (response_text, usage_info)
        """
        try:
            start_time = time.time()

            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config={
                    "temperature": temperature,
                    "top_p": 0.95,
                },
            )

            latency = time.time() - start_time

            # Extract token usage
            usage_info = self._extract_usage(response, call_type)
            usage_info['latency'] = latency

            # Log to wandb if enabled
            if self.use_wandb:
                self._log_to_wandb(sample_id, call_type, usage_info, prompt, response.text, latency, temperature)

            return response.text, usage_info

        except Exception as e:
            logger.error(f"Gemini API call failed: {str(e)}")
            if self.use_wandb:
                wandb.log({
                    f"api_call/{call_type}_error": str(e),
                    "api_call/sample_id": sample_id,
                })
            raise

    def _extract_usage(self, response, call_type: str) -> dict:
        """Extract token usage from response."""
        usage_info = {
            'input_tokens': 0,
            'output_tokens': 0
        }

        if hasattr(response, 'usage_metadata'):
            usage = response.usage_metadata
            usage_info['input_tokens'] = getattr(usage, 'prompt_token_count', 0)
            usage_info['output_tokens'] = getattr(usage, 'candidates_token_count', 0)

            if self.token_counter:
                self.token_counter.add_usage(
                    usage_info['input_tokens'],
                    usage_info['output_tokens'],
                    call_type
                )

        return usage_info

    def _log_to_wandb(self, sample_id, call_type, usage_info, prompt, response_text, latency, temperature):
        """Log API call to wandb."""
        input_cost = (usage_info['input_tokens'] / 1_000_000) * PRICING['input']
        output_cost = (usage_info['output_tokens'] / 1_000_000) * PRICING['output']
        total_cost = input_cost + output_cost

        wandb.log({
            f"api_call/{call_type}_sample_id": sample_id,
            f"api_call/{call_type}_input_tokens": usage_info['input_tokens'],
            f"api_call/{call_type}_output_tokens": usage_info['output_tokens'],
            f"api_call/{call_type}_cost": total_cost,
            f"api_call/{call_type}_latency": latency,
            f"api_call/{call_type}_temperature": temperature,
        })


# =============================================================================
# Data Preprocessing
# =============================================================================

def preprocess_input_data(input_path: str, output_path: str) -> str:
    """
    Preprocess the input data to flatten it so each selected function is one record.

    Args:
        input_path: Path to the merged step 3 output (or single shard)
        output_path: Path to save the flattened data

    Returns:
        Path to the preprocessed file
    """
    logger.info(f"📦 Preprocessing input data from {input_path}")

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    flattened_records = []

    for sample in data:
        sample_id = sample.get('sample_id', '')
        repo_id = sample.get('repo_id', '')
        file_path = sample.get('file_path', '')
        code_content = sample.get('code_content', '')
        code_evaluation = sample.get('code_evaluation', {})

        selected_functions = sample.get('selected_function_list', [])

        for func in selected_functions:
            # Create a unique ID for this function
            function_id = func.get('function_id', 0)
            unique_id = f"{sample_id}_func_{function_id}"

            record = {
                'unique_id': unique_id,
                'sample_id': sample_id,
                'repo_id': repo_id,
                'file_path': file_path,
                'code_content': code_content,
                'code_evaluation': code_evaluation,
                # Function-specific fields
                'function_id': function_id,
                'function_name': func.get('function_name', ''),
                'function_code': func.get('function_code', ''),  # Ground truth
                'masked_code': func.get('masked_code', ''),
                'start_line': func.get('start_line', 0),
                'end_line': func.get('end_line', 0),
                'difficulty_score': func.get('difficulty_score', 0),
                'selection_reason': func.get('selection_reason', ''),
            }

            flattened_records.append(record)

    logger.info(f"📊 Flattened {len(data)} samples into {len(flattened_records)} function records")

    # Save preprocessed data
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(flattened_records, f, ensure_ascii=False, indent=2)

    logger.info(f"💾 Saved preprocessed data to {output_path}")

    return output_path


# =============================================================================
# Response Parsing
# =============================================================================

def parse_fim_response(response: str) -> dict:
    """
    Parse the FIM completion response to extract reasoning and implementation.

    Returns:
        dict with 'reasoning' and 'implementation' keys
    """
    result = {
        'raw_response': response,
        'reasoning': '',
        'implementation': '',
        'parse_success': False
    }

    if not response:
        return result

    # Try to extract reasoning section
    reasoning_match = re.search(
        r'###?\s*Reasoning\s*\n(.*?)(?=###?\s*Implementation|```python)',
        response,
        re.DOTALL | re.IGNORECASE
    )
    if reasoning_match:
        result['reasoning'] = reasoning_match.group(1).strip()

    # Try to extract implementation from code block
    code_match = re.search(
        r'```python\s*\n(.*?)\n```',
        response,
        re.DOTALL
    )
    if code_match:
        result['implementation'] = code_match.group(1).strip()
        result['parse_success'] = True
    else:
        # Fallback: try to find any code after "Implementation"
        impl_match = re.search(
            r'###?\s*Implementation\s*\n(.*)',
            response,
            re.DOTALL | re.IGNORECASE
        )
        if impl_match:
            result['implementation'] = impl_match.group(1).strip()
            result['parse_success'] = True

    return result


def parse_critique_response(response: str) -> dict:
    """
    Parse the critique response to extract scores and reasoning.

    Returns:
        dict with parsed critique data
    """
    result = {
        'raw_response': response,
        'parse_success': False,
        'reasoning': '',
        'scores': {},
        'overall_score': 0,
        'overall_reason': '',
        'key_differences': [],
        'key_similarities': []
    }

    if not response:
        return result

    # Try to extract JSON from code block
    json_match = re.search(r'```json\s*\n(.*?)\n```', response, re.DOTALL)
    if json_match:
        json_str = json_match.group(1).strip()
    else:
        # Try to find JSON object directly
        json_str = extract_json_object(response)

    if not json_str:
        logger.warning("Could not extract JSON from critique response")
        return result

    try:
        parsed = json.loads(json_str)
        result['reasoning'] = parsed.get('reasoning', '')
        result['scores'] = parsed.get('scores', {})
        result['overall_score'] = parsed.get('overall_score', 0)
        result['overall_reason'] = parsed.get('overall_reason', '')
        result['key_differences'] = parsed.get('key_differences', [])
        result['key_similarities'] = parsed.get('key_similarities', [])
        result['parse_success'] = True
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse critique JSON: {e}")

    return result


def extract_json_object(text: str) -> Optional[str]:
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


# =============================================================================
# Main Pipeline
# =============================================================================

class FIMCompletionPipeline:
    """Pipeline for FIM completion and critique."""

    def __init__(
            self,
            input_path: str,
            output_path: str,
            checkpoint_path: str,
            model: str = "gemini-3-flash-preview",
            fim_temperature: float = 0.7,
            critique_temperature: float = 0.3,
            wait_seconds: float = 1.0,
            print_response: bool = False,
            use_wandb: bool = False,
            wandb_project: str = "fim-completion-critique",
            wandb_run_name: str = None,
            shard_id: int = None,
            total_shards: int = None,
            start_idx: int = None,
            end_idx: int = None,
            skip_preprocess: bool = False,
    ):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.model = model
        self.fim_temperature = fim_temperature
        self.critique_temperature = critique_temperature
        self.wait_seconds = wait_seconds
        self.print_response = print_response
        self.skip_preprocess = skip_preprocess

        # Sharding parameters
        self.shard_id = shard_id
        self.total_shards = total_shards
        self.start_idx = start_idx
        self.end_idx = end_idx

        # Token counter
        self.token_counter = TokenCounter()

        # W&B configuration
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name

        # Adjust paths for sharding
        self._adjust_paths_for_sharding()

        # Preprocessed data path
        self.preprocessed_path = self.input_path.parent / f"{self.input_path.stem}_preprocessed.json"

        # Initialize wandb if enabled
        if self.use_wandb:
            self._init_wandb()

        # Initialize Gemini client
        self.gemini_client = GeminiClient(
            model=model,
            token_counter=self.token_counter,
            use_wandb=self.use_wandb
        )

        # Checkpoint data
        self.processed_ids = set()
        self.results = []
        self._load_checkpoint()

    def _adjust_paths_for_sharding(self):
        """Adjust output and checkpoint paths based on sharding configuration."""
        shard_suffix = None

        if self.shard_id is not None and self.total_shards is not None:
            shard_suffix = f"_shard{self.shard_id}_of_{self.total_shards}"
        elif self.start_idx is not None or self.end_idx is not None:
            start = self.start_idx if self.start_idx is not None else 0
            end = self.end_idx if self.end_idx is not None else "end"
            shard_suffix = f"_range_{start}_{end}"

        if shard_suffix:
            output_stem = self.output_path.stem
            output_suffix = self.output_path.suffix
            self.output_path = self.output_path.parent / f"{output_stem}{shard_suffix}{output_suffix}"

            checkpoint_stem = self.checkpoint_path.stem
            checkpoint_suffix = self.checkpoint_path.suffix
            self.checkpoint_path = self.checkpoint_path.parent / f"{checkpoint_stem}{shard_suffix}{checkpoint_suffix}"

            logger.info(f"📂 Adjusted paths for sharding:")
            logger.info(f"   Output: {self.output_path}")
            logger.info(f"   Checkpoint: {self.checkpoint_path}")

    def _get_shard_range(self, total_samples: int) -> tuple[int, int]:
        """Calculate the start and end indices for this shard."""
        if self.start_idx is not None or self.end_idx is not None:
            start = self.start_idx if self.start_idx is not None else 0
            end = self.end_idx if self.end_idx is not None else total_samples
            start = max(0, min(start, total_samples))
            end = max(start, min(end, total_samples))
            return start, end

        if self.shard_id is not None and self.total_shards is not None:
            if self.shard_id < 1 or self.shard_id > self.total_shards:
                raise ValueError(f"shard_id must be between 1 and {self.total_shards}")

            base_size = total_samples // self.total_shards
            remainder = total_samples % self.total_shards

            if self.shard_id <= remainder:
                start = (self.shard_id - 1) * (base_size + 1)
                end = start + base_size + 1
            else:
                start = remainder * (base_size + 1) + (self.shard_id - 1 - remainder) * base_size
                end = start + base_size

            return start, end

        return 0, total_samples

    def _init_wandb(self):
        """Initialize W&B and Weave."""
        logger.info("🔧 Initializing W&B and Weave...")

        base_run_name = self.wandb_run_name or f"fim-critique-{time.strftime('%Y%m%d-%H%M%S')}"

        if self.shard_id is not None and self.total_shards is not None:
            run_name = f"{base_run_name}-shard{self.shard_id}"
        elif self.start_idx is not None or self.end_idx is not None:
            start = self.start_idx if self.start_idx is not None else 0
            end = self.end_idx if self.end_idx is not None else "end"
            run_name = f"{base_run_name}-range{start}-{end}"
        else:
            run_name = base_run_name

        wandb.init(
            project=self.wandb_project,
            name=run_name,
            config={
                "model": self.model,
                "input_path": str(self.input_path),
                "output_path": str(self.output_path),
                "fim_temperature": self.fim_temperature,
                "critique_temperature": self.critique_temperature,
                "wait_seconds": self.wait_seconds,
                "shard_id": self.shard_id,
                "total_shards": self.total_shards,
            },
            resume="allow"
        )

        weave.init(self.wandb_project)
        logger.info(f"✅ W&B initialized: {wandb.run.url}")

    def _load_checkpoint(self):
        """Load checkpoint data if exists."""
        if self.checkpoint_path.exists():
            try:
                with open(self.checkpoint_path, 'r', encoding='utf-8') as f:
                    checkpoint = json.load(f)
                self.processed_ids = set(checkpoint.get('processed_ids', []))
                self.results = checkpoint.get('results', [])

                if 'token_usage' in checkpoint:
                    self.token_counter.load_from_dict(checkpoint['token_usage'])

                logger.info(f"📂 Loaded checkpoint:")
                logger.info(f"   Processed records: {len(self.processed_ids)}")
                logger.info(f"   Results saved: {len(self.results)}")

                cost = self.token_counter.get_cost()
                logger.info(f"   Total cost so far: ${cost['total_cost']:.4f}")

            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")

    def _save_checkpoint(self):
        """Save current progress to checkpoint file."""
        checkpoint = {
            'processed_ids': list(self.processed_ids),
            'results': self.results,
            'token_usage': self.token_counter.to_dict()
        }

        temp_path = self.checkpoint_path.with_suffix('.tmp')
        try:
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(checkpoint, f, ensure_ascii=False, indent=2)
            temp_path.replace(self.checkpoint_path)
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            if temp_path.exists():
                temp_path.unlink()

    def _print_response(self, title: str, unique_id: str, response: str, usage_info: dict):
        """Print API response."""
        print("\n" + "=" * 80)
        print(f"📝 {title} for: {unique_id}")
        print(f"   Tokens: input={usage_info['input_tokens']:,}, output={usage_info['output_tokens']:,}")
        print(f"   Latency: {usage_info.get('latency', 0):.2f}s")
        print("=" * 80)
        print(response[:2000] + "..." if len(response) > 2000 else response)
        print("=" * 80 + "\n")

    def _print_current_cost(self):
        """Print current token usage and cost."""
        cost = self.token_counter.get_cost()
        print(f"💰 累计费用: ${cost['total_cost']:.4f} "
              f"(输入: {cost['input_tokens']:,}, 输出: {cost['output_tokens']:,}, "
              f"FIM: {cost['fim_requests']}, Critique: {cost['critique_requests']})")

    def process_single_record(self, record: dict) -> Optional[dict]:
        """
        Process a single function record: FIM completion + Critique.

        Returns:
            The record with added completion and critique data
        """
        unique_id = record['unique_id']
        function_name = record['function_name']
        masked_code = record['masked_code']
        ground_truth = record['function_code']

        result = record.copy()

        # =====================================================================
        # Step 1: FIM Completion
        # =====================================================================
        fim_prompt = FIM_COMPLETION_PROMPT.format(
            masked_code=masked_code,
            function_name=function_name
        )

        try:
            fim_response, fim_usage = self.gemini_client.get_response(
                prompt=fim_prompt,
                sample_id=unique_id,
                call_type="fim",
                temperature=self.fim_temperature
            )

            if self.print_response:
                self._print_response("FIM Completion", unique_id, fim_response, fim_usage)

            # Parse FIM response
            fim_parsed = parse_fim_response(fim_response)

            result['fim_response'] = {
                'raw_response': fim_response,
                'reasoning': fim_parsed['reasoning'],
                'implementation': fim_parsed['implementation'],
                'parse_success': fim_parsed['parse_success'],
                'token_usage': fim_usage
            }

            if self.wait_seconds > 0:
                time.sleep(self.wait_seconds)

        except Exception as e:
            logger.error(f"FIM completion failed for {unique_id}: {e}")
            result['fim_response'] = {
                'error': str(e),
                'parse_success': False
            }
            return result

        # =====================================================================
        # Step 2: Critique
        # =====================================================================
        completion = fim_parsed['implementation'] if fim_parsed['parse_success'] else fim_response

        critique_prompt = CRITIQUE_PROMPT.format(
            masked_code=masked_code,
            function_name=function_name,
            ground_truth=ground_truth,
            completion=completion
        )

        try:
            critique_response, critique_usage = self.gemini_client.get_response(
                prompt=critique_prompt,
                sample_id=unique_id,
                call_type="critique",
                temperature=self.critique_temperature
            )

            if self.print_response:
                self._print_response("Critique", unique_id, critique_response, critique_usage)

            # Parse critique response
            critique_parsed = parse_critique_response(critique_response)

            result['critique_response'] = {
                'raw_response': critique_response,
                'reasoning': critique_parsed['reasoning'],
                'scores': critique_parsed['scores'],
                'overall_score': critique_parsed['overall_score'],
                'overall_reason': critique_parsed['overall_reason'],
                'key_differences': critique_parsed['key_differences'],
                'key_similarities': critique_parsed['key_similarities'],
                'parse_success': critique_parsed['parse_success'],
                'token_usage': critique_usage
            }

            if self.wait_seconds > 0:
                time.sleep(self.wait_seconds)

        except Exception as e:
            logger.error(f"Critique failed for {unique_id}: {e}")
            result['critique_response'] = {
                'error': str(e),
                'parse_success': False
            }

        # Log to wandb
        if self.use_wandb and result.get('critique_response', {}).get('parse_success'):
            wandb.log({
                "scores/overall": result['critique_response']['overall_score'],
                "scores/correctness": result['critique_response']['scores'].get('correctness', 0),
                "scores/executability": result['critique_response']['scores'].get('executability', 0),
                "scores/api_usage": result['critique_response']['scores'].get('api_usage', 0),
                "scores/readability": result['critique_response']['scores'].get('readability', 0),
                "scores/completeness": result['critique_response']['scores'].get('completeness', 0),
            })

        return result

    def run(self):
        """Run the FIM completion and critique pipeline."""
        # Preprocess data if needed
        if not self.skip_preprocess or not self.preprocessed_path.exists():
            preprocess_input_data(str(self.input_path), str(self.preprocessed_path))

        # Load preprocessed data
        logger.info(f"📂 Loading preprocessed data from {self.preprocessed_path}")
        with open(self.preprocessed_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        total_records_in_file = len(data)

        # Get shard range
        shard_start, shard_end = self._get_shard_range(total_records_in_file)
        data = data[shard_start:shard_end]
        total_records = len(data)

        logger.info(f"📊 Sharding info:")
        logger.info(f"   Total records in file: {total_records_in_file}")
        logger.info(f"   This shard's range: [{shard_start}, {shard_end})")
        logger.info(f"   Records in this shard: {total_records}")

        # Filter out processed records
        remaining_records = [r for r in data if r['unique_id'] not in self.processed_ids]

        logger.info(f"   Already processed: {len(self.processed_ids)}")
        logger.info(f"   Remaining: {len(remaining_records)}")

        self._print_current_cost()

        # Build shard prefix
        shard_prefix = ""
        if self.shard_id is not None:
            shard_prefix = f"[Shard {self.shard_id}/{self.total_shards}] "

        # Process records
        with tqdm(total=len(remaining_records), desc=f"{shard_prefix}Processing") as pbar:
            for record in remaining_records:
                unique_id = record['unique_id']

                # Update progress
                cost = self.token_counter.get_cost()
                pbar.set_description(
                    f"{shard_prefix}Processed: {len(self.processed_ids)} | "
                    f"Cost: ${cost['total_cost']:.3f}"
                )

                try:
                    result = self.process_single_record(record)
                    if result:
                        self.results.append(result)
                        self.processed_ids.add(unique_id)

                        # Log status
                        overall_score = result.get('critique_response', {}).get('overall_score', 'N/A')
                        logger.info(f"  ✅ {unique_id}: Overall score = {overall_score}")

                except Exception as e:
                    logger.error(f"Error processing {unique_id}: {e}")
                    self.processed_ids.add(unique_id)

                # Save checkpoint
                self._save_checkpoint()
                self._print_current_cost()

                pbar.update(1)

        # Save final output
        logger.info(f"\n💾 Saving {len(self.results)} results to {self.output_path}")
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)

        # Print summary
        self._print_summary()

        if self.use_wandb:
            wandb.finish()

    def _print_summary(self):
        """Print processing summary."""
        print("\n" + "=" * 60)
        print("📊 Processing Summary")
        if self.shard_id is not None:
            print(f"   Shard: {self.shard_id}/{self.total_shards}")
        print("=" * 60)

        print(f"\n📈 Statistics:")
        print(f"   Total processed: {len(self.processed_ids)}")
        print(f"   Results saved: {len(self.results)}")

        self.token_counter.print_cost_summary()

        if self.results:
            # Score distribution
            scores = [r.get('critique_response', {}).get('overall_score', 0)
                      for r in self.results
                      if r.get('critique_response', {}).get('parse_success')]

            if scores:
                avg_score = sum(scores) / len(scores)
                print(f"\n📊 Score Statistics:")
                print(f"   Average overall score: {avg_score:.2f}")

                score_dist = {}
                for s in scores:
                    score_dist[s] = score_dist.get(s, 0) + 1
                print(f"   Score distribution:")
                for score in sorted(score_dist.keys()):
                    print(f"      Score {score}: {score_dist[score]} ({score_dist[score] / len(scores) * 100:.1f}%)")

            # Parse success rate
            fim_success = sum(1 for r in self.results if r.get('fim_response', {}).get('parse_success'))
            critique_success = sum(1 for r in self.results if r.get('critique_response', {}).get('parse_success'))
            print(f"\n📋 Parse Success Rate:")
            print(f"   FIM: {fim_success}/{len(self.results)} ({fim_success / len(self.results) * 100:.1f}%)")
            print(
                f"   Critique: {critique_success}/{len(self.results)} ({critique_success / len(self.results) * 100:.1f}%)")

        print("\n" + "=" * 60)


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="FIM Completion and Critique Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Single process (all data):
  python step_4_fim_completion_and_critique.py -i input.json -o output.json

  # Parallel with 50 shards:
  for i in $(seq 1 50); do
    nohup python step_4_fim_completion_and_critique.py \\
      -i input.json -o output.json \\
      --shard $i --total-shards 50 \\
      --wandb --wandb-run-name "exp_1" \\
      > shard_$i.log 2>&1 &
  done

  # Or with explicit range:
  python step_4_fim_completion_and_critique.py --start-idx 0 --end-idx 100
        '''
    )

    # Input/Output
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to input JSON file (step 3 output or merged file)"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Path to output JSON file"
    )
    parser.add_argument(
        "--checkpoint", "-c",
        default=None,
        help="Path to checkpoint file (default: output_path with _checkpoint suffix)"
    )

    # Model configuration
    parser.add_argument(
        "--model", "-m",
        default="gemini-3-flash-preview",
        help="Gemini model to use"
    )
    parser.add_argument(
        "--fim-temperature",
        type=float,
        default=0.7,
        help="Temperature for FIM completion (default: 0.7)"
    )
    parser.add_argument(
        "--critique-temperature",
        type=float,
        default=0.3,
        help="Temperature for critique (default: 0.3)"
    )

    # Processing options
    parser.add_argument(
        "--wait", "-w",
        type=float,
        default=1.0,
        help="Seconds to wait between API calls (default: 1.0)"
    )
    parser.add_argument(
        "--print-response", "-p",
        action="store_true",
        help="Print API responses"
    )
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Skip preprocessing if preprocessed file exists"
    )

    # Sharding
    parser.add_argument(
        "--shard",
        type=int,
        default=None,
        help="Shard ID (1-indexed). Use with --total-shards"
    )
    parser.add_argument(
        "--total-shards",
        type=int,
        default=None,
        help="Total number of shards. Use with --shard"
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=None,
        help="Start index (0-indexed, inclusive)"
    )
    parser.add_argument(
        "--end-idx",
        type=int,
        default=None,
        help="End index (0-indexed, exclusive)"
    )

    # W&B
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable W&B logging"
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="fim-completion-critique-1230",
        help="W&B project name"
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="W&B run name"
    )

    args = parser.parse_args()

    # Default checkpoint path
    if args.checkpoint is None:
        output_path = Path(args.output)
        args.checkpoint = str(output_path.parent / f"{output_path.stem}_checkpoint.json")

    # Validate sharding arguments
    if args.shard is not None and args.total_shards is None:
        parser.error("--shard requires --total-shards")
    if args.total_shards is not None and args.shard is None:
        parser.error("--total-shards requires --shard")
    if args.shard is not None and (args.start_idx is not None or args.end_idx is not None):
        parser.error("Cannot use --shard with --start-idx or --end-idx")

    # Check wandb
    if args.wandb and not WANDB_AVAILABLE:
        logger.warning("⚠️ W&B requested but not available")
        args.wandb = False

    # Run pipeline
    pipeline = FIMCompletionPipeline(
        input_path=args.input,
        output_path=args.output,
        checkpoint_path=args.checkpoint,
        model=args.model,
        fim_temperature=args.fim_temperature,
        critique_temperature=args.critique_temperature,
        wait_seconds=args.wait,
        print_response=args.print_response,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        shard_id=args.shard,
        total_shards=args.total_shards,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        skip_preprocess=args.skip_preprocess,
    )

    pipeline.run()


if __name__ == "__main__":
    main()
    """
    for i in $(seq 1 50); do
  nohup python step_4_fim_completion_and_critique.py \
    -i step_3_output.json \
    -o step_4_output.json \
    --shard $i --total-shards 50 \
    --wandb --wandb-run-name "exp_v1" \
    > logs/shard_$i.log 2>&1 &
done
    """