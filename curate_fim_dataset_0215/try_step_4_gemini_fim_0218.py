#!/usr/bin/env python3
"""
FIM Completion and Critique Pipeline (v2)

This script processes the selected functions and:
1. Calls Gemini to complete the masked function (with CoT reasoning)
2. Calls Gemini again to critique the completion against ground truth (with CoT),
   including a feasibility judgment on whether the function is suitable for FIM evaluation.

Supports sharding for parallel processing and checkpoint-based resumption.
Supports --guided mode: passes ground truth to FIM completion for higher quality,
while instructing the model to write reasoning as independent context analysis.

Input format: A JSON file containing a list of dicts, where each dict represents
one function record with fields like sample_id, func_name, func_content (ground truth),
masked_code_content, etc.
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
# Unique ID Generation (single source of truth)
# =============================================================================

def make_unique_id(record: dict) -> str:
    """
    Generate a deterministic unique ID for a record.

    Uses sample_id + func_name + start_line to ensure uniqueness.
    This is a module-level function so that run() and process_single_record()
    always produce the same ID for the same record — critical for checkpoint
    consistency.
    """
    sample_id = record.get('sample_id', '')
    func_name = record.get('func_name', '')
    start_line = record.get('start_line', 0)
    return f"{sample_id}_{func_name}_{start_line}"


# =============================================================================
# Prompt Templates
# =============================================================================

FIM_COMPLETION_PROMPT = '''You are an expert Python programmer. Your task is to complete a masked function based on the surrounding code context.

## Task

Below is a Python file where one function's body has been replaced with `# <MASKED_FUNCTION_BODY>`. Your job is to:

1. **Analyze the context**: Look at the imports, other functions, class definitions, and how this function is called or calls other functions.
2. **Reason step by step**: Think through what this function should do based on:
   - The function signature (name, parameters, type hints)
   - The docstring (if visible in the signature area)
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

FIM_COMPLETION_PROMPT_GUIDED = '''You are an expert Python programmer. Your task is to complete a masked function based on the surrounding code context.

## Task

Below is a Python file where one function's body has been replaced with `# <MASKED_FUNCTION_BODY>`. Your job is to:

1. **Analyze the context**: Look at the imports, other functions, class definitions, and how this function is called or calls other functions.
2. **Reason step by step**: Think through what this function should do based on:
   - The function signature (name, parameters, type hints)
   - The docstring (if visible in the signature area)
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

## Critical Instructions for Reasoning
- In the Reasoning section, you must ONLY describe your analysis of the code context: the function signature, how it is used, what patterns you observe, and what the function logically should do.
- Your reasoning must read as a natural, independent chain of thought — as if you are figuring out the implementation purely from the surrounding code.
- Do NOT mention, reference, or hint at any "reference", "hint", "example", "ground truth", or "provided implementation" in the Reasoning section. Write as if you are discovering the logic yourself.
- The Reasoning section should demonstrate genuine code comprehension: explain WHY each part of the implementation makes sense given the context.

## Critical Instructions for Implementation
- You are given a reference implementation below to help you understand the function's PURPOSE, LOGIC, and EDGE CASES.
- However, you must write YOUR OWN version of the implementation. Do NOT copy the reference line by line.
- Your implementation must be **functionally equivalent** (same inputs → same outputs, same side effects) but should reflect your own coding choices:
  - You may use different variable names
  - You may restructure the control flow (e.g., early returns vs. if-else chains, list comprehensions vs. loops)
  - You may reorder independent statements
  - You may use different but equivalent standard library calls
  - You may add or adjust comments in your own words
- The goal is: a competent programmer who understands the same requirements would write it this way — not a copy-paste of the reference.
- If the reference implementation is very short or trivial (e.g., a single return statement, a one-liner), it is acceptable for your implementation to look very similar, since there is essentially only one natural way to write it.

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

## Reference Implementation (for understanding intent only — do NOT copy verbatim)

Study this implementation to understand what the function should accomplish, what edge cases it handles, and what APIs it uses. Then write your own version that achieves the same behavior.

```python
{ground_truth}
```

Please analyze the context and provide your reasoning followed by the implementation.
'''

CRITIQUE_PROMPT = '''You are an expert code reviewer. Your task is to evaluate a function completion against the ground truth implementation, AND determine whether this function is suitable for FIM (Fill-in-the-Middle) evaluation.

## Task

A model was asked to complete a masked function based on surrounding code context. You need to:
1. **Feasibility Check**: First, determine whether this function is actually feasible to complete based solely on the code context provided. Some functions rely on obscure external libraries, domain-specific APIs, or conventions that cannot be reasonably inferred from the surrounding code. These functions should be marked as infeasible.
2. **Quality Evaluation**: Compare the completion with the ground truth and score the completion across multiple dimensions.

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

## Part 1: Feasibility Assessment

Determine whether this function is a FAIR test for FIM completion. A function is **infeasible** if:
- It relies heavily on **uncommon or domain-specific external libraries/APIs** whose usage patterns cannot be inferred from the visible context (e.g., calling specific methods of a niche library that isn't used elsewhere in the file).
- It requires **knowledge of external conventions, configurations, or data formats** not present in the file.
- The surrounding context provides **insufficient information** to reconstruct the core logic (e.g., the function implements an algorithm whose details are not hinted at anywhere in the file).
- It uses **magic constants, hardcoded strings, or lookup tables** that are specific to external systems and cannot be guessed.

A function is **feasible** if:
- Its behavior can be reasonably inferred from the function name, signature, docstring, and surrounding code.
- It follows patterns established by other functions in the file.
- It uses standard Python or well-known library APIs that are already imported/used in the file.
- Even if the exact implementation differs from ground truth, a competent programmer could write a functionally equivalent version from the available context.

## Part 2: Quality Evaluation Criteria

Evaluate the completion on these dimensions (1-5 scale, where 5 is best):

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
  "feasibility": {{
    "is_feasible": <true or false>,
    "confidence": <0.0 to 1.0, how confident you are in this judgment>,
    "reason": "<detailed explanation of why this function is or isn't feasible for FIM evaluation>",
    "infeasible_factors": ["<list of specific factors making it infeasible, empty if feasible>"]
  }},
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
  "key_similarities": ["<list of main similarities>"],
  "should_discard": <true or false>,
  "discard_reason": "<if should_discard is true, explain why this sample should be discarded from the dataset>"
}}
```

## Decision Logic for `should_discard`

Set `should_discard` to `true` if ANY of the following conditions are met:
1. **Infeasible**: `is_feasible` is `false` — the function fundamentally cannot be completed from context alone.
2. **Very Low Quality**: `overall_score` <= 2 AND `executability` <= 2 — the model's completion is too poor to be useful as training/evaluation data.
3. **Non-executable Garbage**: `executability` == 1 — the completion is completely non-functional.

Otherwise, set `should_discard` to `false`. Even if the completion isn't perfect, it can still be valuable data if the function is feasible and the completion is at least partially correct or executable.

## Scoring Guidelines for Overall Score
- **5**: The completion is functionally equivalent to the ground truth. It would work correctly as a drop-in replacement.
- **4**: The completion is mostly correct with minor issues that wouldn't significantly impact functionality.
- **3**: The completion captures the main idea but has notable differences or issues.
- **2**: The completion has major problems but shows some understanding of the task.
- **1**: The completion is fundamentally incorrect or completely misses the point.

Note: A high correctness score but low executability score indicates the model understood the logic but made implementation mistakes. This distinction is important for training data quality assessment.

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
        self.fim_input_tokens = 0
        self.fim_output_tokens = 0
        self.fim_requests = 0
        self.critique_input_tokens = 0
        self.critique_output_tokens = 0
        self.critique_requests = 0

    def add_usage(self, input_tokens: int, output_tokens: int, call_type: str = "fim"):
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
        cost = self.get_cost()
        print(f"\n💰 Token Usage & Cost:")
        print(f"   Input tokens:  {cost['input_tokens']:,} (${cost['input_cost']:.4f})")
        print(f"   Output tokens: {cost['output_tokens']:,} (${cost['output_cost']:.4f})")
        print(f"   Total tokens:  {cost['total_tokens']:,}")
        print(f"   FIM requests:  {cost['fim_requests']}")
        print(f"   Critique requests: {cost['critique_requests']}")
        print(f"   💵 Total cost: ${cost['total_cost']:.4f}")

    def to_dict(self) -> dict:
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
            temperature: float = 0.7,
            max_retries: int = 3,
            retry_delay: float = 5.0,
    ) -> tuple[str, dict]:
        """
        Get response from Gemini API with retry logic.

        Args:
            prompt: The prompt to send
            sample_id: Identifier for logging
            call_type: "fim" or "critique"
            temperature: Sampling temperature
            max_retries: Maximum number of retries on failure
            retry_delay: Base delay between retries (exponential backoff)

        Returns:
            tuple: (response_text, usage_info)
        """
        last_error = None
        for attempt in range(max_retries):
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

                usage_info = self._extract_usage(response, call_type)
                usage_info['latency'] = latency

                if self.use_wandb:
                    self._log_to_wandb(sample_id, call_type, usage_info, prompt, response.text, latency, temperature)

                return response.text, usage_info

            except Exception as e:
                last_error = e
                wait = retry_delay * (2 ** attempt)
                logger.warning(f"Gemini API call failed (attempt {attempt + 1}/{max_retries}): {e}. "
                               f"Retrying in {wait:.1f}s...")
                if attempt < max_retries - 1:
                    time.sleep(wait)

        logger.error(f"Gemini API call failed after {max_retries} attempts: {last_error}")
        if self.use_wandb:
            wandb.log({
                f"api_call/{call_type}_error": str(last_error),
                "api_call/sample_id": sample_id,
            })
        raise last_error

    def _extract_usage(self, response, call_type: str) -> dict:
        usage_info = {'input_tokens': 0, 'output_tokens': 0}
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
# Response Parsing
# =============================================================================

def parse_fim_response(response: str) -> dict:
    """Parse the FIM completion response to extract reasoning and implementation."""
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
    """Parse the critique response to extract feasibility, scores, and reasoning."""
    result = {
        'raw_response': response,
        'parse_success': False,
        # Feasibility fields
        'feasibility': {
            'is_feasible': True,
            'confidence': 0.0,
            'reason': '',
            'infeasible_factors': []
        },
        # Quality fields
        'reasoning': '',
        'scores': {},
        'overall_score': 0,
        'overall_reason': '',
        'key_differences': [],
        'key_similarities': [],
        # Discard decision
        'should_discard': False,
        'discard_reason': ''
    }

    if not response:
        return result

    # Try to extract JSON from code block
    json_match = re.search(r'```json\s*\n(.*?)\n```', response, re.DOTALL)
    if json_match:
        json_str = json_match.group(1).strip()
    else:
        json_str = extract_json_object(response)

    if not json_str:
        logger.warning("Could not extract JSON from critique response")
        return result

    try:
        parsed = json.loads(json_str)

        # Parse feasibility
        feasibility = parsed.get('feasibility', {})
        result['feasibility'] = {
            'is_feasible': feasibility.get('is_feasible', True),
            'confidence': feasibility.get('confidence', 0.0),
            'reason': feasibility.get('reason', ''),
            'infeasible_factors': feasibility.get('infeasible_factors', [])
        }

        # Parse quality scores
        result['reasoning'] = parsed.get('reasoning', '')
        result['scores'] = parsed.get('scores', {})
        result['overall_score'] = parsed.get('overall_score', 0)
        result['overall_reason'] = parsed.get('overall_reason', '')
        result['key_differences'] = parsed.get('key_differences', [])
        result['key_similarities'] = parsed.get('key_similarities', [])

        # Parse discard decision
        result['should_discard'] = parsed.get('should_discard', False)
        result['discard_reason'] = parsed.get('discard_reason', '')

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

        if char == '"':
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
            guided: bool = False,
            use_wandb: bool = False,
            wandb_project: str = "fim-completion-critique",
            wandb_run_name: str = None,
            shard_id: int = None,
            total_shards: int = None,
            start_idx: int = None,
            end_idx: int = None,
    ):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.model = model
        self.fim_temperature = fim_temperature
        self.critique_temperature = critique_temperature
        self.wait_seconds = wait_seconds
        self.print_response = print_response
        self.guided = guided

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

        # Adjust paths for sharding (must happen before _load_checkpoint)
        self._adjust_paths_for_sharding()

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

        if self.guided:
            logger.info("🎯 Guided mode ENABLED: FIM completion will use ground truth reference")

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
                "guided": self.guided,
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
        """Save current progress to checkpoint file (atomic write via temp file)."""
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
        """Print API response for debugging."""
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
        print(f"💰 Cumulative cost: ${cost['total_cost']:.4f} "
              f"(in: {cost['input_tokens']:,}, out: {cost['output_tokens']:,}, "
              f"FIM: {cost['fim_requests']}, Critique: {cost['critique_requests']})")

    def process_single_record(self, record: dict) -> Optional[dict]:
        """
        Process a single function record: FIM completion + Critique.

        Returns:
            The record with added completion, critique, and discard decision.
        """
        unique_id = make_unique_id(record)  # Module-level function — same as run()
        function_name = record.get('func_name', '')
        masked_code = record.get('masked_code_content', '')
        ground_truth = record.get('func_content', '')

        # Start with all original fields preserved
        result = record.copy()
        result['unique_id'] = unique_id
        result['guided_mode'] = self.guided

        # =====================================================================
        # Step 1: FIM Completion
        # =====================================================================
        if self.guided:
            # Guided mode: include ground truth as hidden reference
            fim_prompt = FIM_COMPLETION_PROMPT_GUIDED.format(
                masked_code=masked_code,
                function_name=function_name,
                ground_truth=ground_truth
            )
        else:
            # Standard mode: no ground truth
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
            result['should_discard'] = None  # Unknown — API error
            result['discard_reason'] = f"FIM API call failed: {e}"
            result['is_feasible'] = None  # Unknown — never evaluated
            return result

        # =====================================================================
        # Step 2: Critique (includes feasibility check)
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

            critique_parsed = parse_critique_response(critique_response)

            result['critique_response'] = {
                'raw_response': critique_response,
                'feasibility': critique_parsed['feasibility'],
                'reasoning': critique_parsed['reasoning'],
                'scores': critique_parsed['scores'],
                'overall_score': critique_parsed['overall_score'],
                'overall_reason': critique_parsed['overall_reason'],
                'key_differences': critique_parsed['key_differences'],
                'key_similarities': critique_parsed['key_similarities'],
                'should_discard': critique_parsed['should_discard'],
                'discard_reason': critique_parsed['discard_reason'],
                'parse_success': critique_parsed['parse_success'],
                'token_usage': critique_usage
            }

            # Promote discard decision to top level for easy filtering
            result['should_discard'] = critique_parsed['should_discard']
            result['discard_reason'] = critique_parsed['discard_reason']
            result['is_feasible'] = critique_parsed['feasibility']['is_feasible']

            if self.wait_seconds > 0:
                time.sleep(self.wait_seconds)

        except Exception as e:
            logger.error(f"Critique failed for {unique_id}: {e}")
            result['critique_response'] = {
                'error': str(e),
                'parse_success': False
            }
            result['should_discard'] = None  # Unknown — API error
            result['discard_reason'] = f"Critique API call failed: {e}"
            result['is_feasible'] = None  # Unknown — never evaluated

        # Log to wandb
        if self.use_wandb and result.get('critique_response', {}).get('parse_success'):
            cr = result['critique_response']
            wandb.log({
                "scores/overall": cr['overall_score'],
                "scores/correctness": cr['scores'].get('correctness', 0),
                "scores/executability": cr['scores'].get('executability', 0),
                "scores/api_usage": cr['scores'].get('api_usage', 0),
                "scores/readability": cr['scores'].get('readability', 0),
                "scores/completeness": cr['scores'].get('completeness', 0),
                "feasibility/is_feasible": cr['feasibility']['is_feasible'],
                "feasibility/confidence": cr['feasibility']['confidence'],
                "decision/should_discard": cr['should_discard'],
            })

        return result

    def run(self):
        """Run the FIM completion and critique pipeline."""
        # Load input data directly (no preprocessing needed)
        logger.info(f"📂 Loading input data from {self.input_path}")
        with open(self.input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        total_records_in_file = len(data)

        # Get shard range
        shard_start, shard_end = self._get_shard_range(total_records_in_file)
        data = data[shard_start:shard_end]

        logger.info(f"📊 Sharding info:")
        logger.info(f"   Total records in file: {total_records_in_file}")
        logger.info(f"   This shard's range: [{shard_start}, {shard_end})")
        logger.info(f"   Records in this shard: {len(data)}")
        logger.info(f"   Guided mode: {self.guided}")

        # Generate unique IDs using the SAME module-level function
        # that process_single_record() uses — ensures checkpoint consistency.
        for record in data:
            record['_unique_id'] = make_unique_id(record)

        remaining_records = [r for r in data if r['_unique_id'] not in self.processed_ids]

        logger.info(f"   Already processed: {len(self.processed_ids)}")
        logger.info(f"   Remaining: {len(remaining_records)}")

        self._print_current_cost()

        # Build shard prefix for progress bar
        shard_prefix = ""
        if self.shard_id is not None:
            shard_prefix = f"[Shard {self.shard_id}/{self.total_shards}] "

        # Process records
        with tqdm(total=len(remaining_records), desc=f"{shard_prefix}Processing") as pbar:
            for record in remaining_records:
                unique_id = record['_unique_id']

                # Update progress bar
                cost = self.token_counter.get_cost()
                pbar.set_description(
                    f"{shard_prefix}Processed: {len(self.processed_ids)} | "
                    f"Cost: ${cost['total_cost']:.3f}"
                )

                try:
                    # Remove temp key before processing
                    record_clean = {k: v for k, v in record.items() if k != '_unique_id'}
                    result = self.process_single_record(record_clean)

                    if result:
                        self.results.append(result)
                        self.processed_ids.add(unique_id)

                        # Log status
                        overall_score = result.get('critique_response', {}).get('overall_score', 'N/A')
                        is_feasible = result.get('is_feasible', 'N/A')
                        should_discard = result.get('should_discard', 'N/A')
                        logger.info(f"  ✅ {unique_id}: score={overall_score}, "
                                    f"feasible={is_feasible}, discard={should_discard}")

                except Exception as e:
                    logger.error(f"Error processing {unique_id}: {e}")
                    self.processed_ids.add(unique_id)

                # Save checkpoint after every record
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
        """Print processing summary with feasibility and discard statistics."""
        print("\n" + "=" * 60)
        print("📊 Processing Summary")
        if self.shard_id is not None:
            print(f"   Shard: {self.shard_id}/{self.total_shards}")
        if self.guided:
            print(f"   Mode: GUIDED (ground truth visible to FIM)")
        print("=" * 60)

        print(f"\n📈 Statistics:")
        print(f"   Total processed: {len(self.processed_ids)}")
        print(f"   Results saved: {len(self.results)}")

        self.token_counter.print_cost_summary()

        if self.results:
            # Feasibility stats (only count records where feasibility was evaluated)
            feasible_count = sum(1 for r in self.results if r.get('is_feasible') is True)
            infeasible_count = sum(1 for r in self.results if r.get('is_feasible') is False)
            unknown_feasibility = sum(1 for r in self.results if r.get('is_feasible') is None)
            evaluated = feasible_count + infeasible_count
            print(f"\n🔍 Feasibility:")
            if evaluated > 0:
                print(f"   Feasible: {feasible_count} ({feasible_count / evaluated * 100:.1f}%)")
                print(f"   Infeasible: {infeasible_count} ({infeasible_count / evaluated * 100:.1f}%)")
            if unknown_feasibility:
                print(f"   Unknown (API error): {unknown_feasibility}")

            # Discard stats
            discard_count = sum(1 for r in self.results if r.get('should_discard') is True)
            keep_count = sum(1 for r in self.results if r.get('should_discard') is False)
            unknown_count = sum(1 for r in self.results if r.get('should_discard') is None)
            print(f"\n🗑️ Discard Decision:")
            print(f"   Keep: {keep_count}")
            print(f"   Discard: {discard_count}")
            if unknown_count:
                print(f"   Unknown (API error): {unknown_count}")

            # Score distribution (only for successfully parsed critiques)
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
            n = len(self.results)
            print(f"\n📋 Parse Success Rate:")
            print(f"   FIM: {fim_success}/{n} ({fim_success / n * 100:.1f}%)")
            print(f"   Critique: {critique_success}/{n} ({critique_success / n * 100:.1f}%)")

        print("\n" + "=" * 60)


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="FIM Completion and Critique Pipeline (v2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Standard mode (no ground truth in FIM):
  python step_4_gemini_fim_and_critique_0217.py -i input.json -o output.json

  # Guided mode (ground truth visible to FIM for higher quality):
  python step_4_gemini_fim_and_critique_0217.py -i input.json -o output.json --guided

  # Parallel with 50 shards:
  for i in $(seq 1 50); do
    nohup python step_4_gemini_fim_and_critique_0217.py \\
      -i input.json -o output.json --guided \\
      --shard $i --total-shards 50 \\
      --wandb --wandb-run-name "exp_1" \\
      > shard_$i.log 2>&1 &
  done

  # Or with explicit range:
  python step_4_gemini_fim_and_critique_0217.py --start-idx 0 --end-idx 100 --guided
        '''
    )

    # Input/Output
    parser.add_argument("--input", "-i", required=True, help="Path to input JSON file (flat list of function records)")
    parser.add_argument("--output", "-o", required=True, help="Path to output JSON file")
    parser.add_argument("--checkpoint", "-c", default=None, help="Path to checkpoint file (default: auto-generated)")

    # Model configuration
    parser.add_argument("--model", "-m", default="gemini-3-flash-preview", help="Gemini model to use")
    parser.add_argument("--fim-temperature", type=float, default=0.7, help="Temperature for FIM completion (default: 0.7)")
    parser.add_argument("--critique-temperature", type=float, default=0.3, help="Temperature for critique (default: 0.3)")

    # Processing options
    parser.add_argument("--wait", "-w", type=float, default=0.5, help="Seconds to wait between API calls (default: 0.5)")
    parser.add_argument("--print-response", "-p", action="store_true", help="Print API responses")
    parser.add_argument("--guided", "-g", action="store_true",
                        help="Guided mode: pass ground truth to FIM completion as hidden reference. "
                             "The model sees the answer but must write reasoning as independent analysis.")

    # Sharding
    parser.add_argument("--shard", type=int, default=None, help="Shard ID (1-indexed). Use with --total-shards")
    parser.add_argument("--total-shards", type=int, default=None, help="Total number of shards")
    parser.add_argument("--start-idx", type=int, default=None, help="Start index (0-indexed, inclusive)")
    parser.add_argument("--end-idx", type=int, default=None, help="End index (0-indexed, exclusive)")

    # W&B
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--wandb-project", type=str, default="fim-completion-critique", help="W&B project name")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="W&B run name")

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
        guided=args.guided,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        shard_id=args.shard,
        total_shards=args.total_shards,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
    )

    pipeline.run()


if __name__ == "__main__":
    main()