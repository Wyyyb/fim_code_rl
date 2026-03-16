#!/usr/bin/env python3
"""
Multi-Function FIM Completion and Critique Pipeline (v2)

Extension of the single-function FIM pipeline for groups of 2-3 related
functions. Uses Gemini to complete multiple masked functions simultaneously,
then critiques each function individually with an additional group-coherence
assessment.

═══════════════════════════════════════════════════════════════════════
Key differences from single-function pipeline:
═══════════════════════════════════════════════════════════════════════
1. Input: reads from *_groups.json (output of depfim_multi.py)
2. FIM prompt: asks the model to complete ALL masked functions together,
   emphasizing cross-function consistency (interface contracts, shared
   state, call-chain coherence)
3. Critique prompt: evaluates each function individually + a group-level
   coherence score. If ANY function scores ≤2 on executability or the
   group coherence score is ≤2, the entire group is discarded.
4. Pre-sharding: the large input file is split into per-shard files on
   disk ONCE, so parallel workers only read their own small shard file
   (avoids OOM from all workers loading the full dataset).
5. Separate output: 2-function and 3-function results are saved to
   different files for easy downstream use.
6. New output fields: group_size, group_type, coupling, group_score,
   group_difficulty, per-function critique scores, group_coherence_score.
7. Guided mode (--guided): provides ground truth to the model for
   higher-quality reasoning, while requiring the reasoning to read as
   independent analysis (no mention of references or hints).

═══════════════════════════════════════════════════════════════════════
Usage:
═══════════════════════════════════════════════════════════════════════

  # Step 0 (run once): pre-split the input into shard files
  python step_4_multi_fim_gemini.py --pre-shard \\
    -i /data/.../step_3_selected_multi_fim_functions_0227_groups.json \\
    --total-shards 50

  # Step 1: run each shard in parallel (with guided mode)
  for i in $(seq 1 50); do
    nohup python step_4_multi_fim_gemini.py \\
      -i /data/.../step_3_selected_multi_fim_functions_0227_groups.json \\
      -o /data/.../step_4_multi_fim_output_0227.json \\
      --shard $i --total-shards 50 --guided \\
      > shard_$i.log 2>&1 &
  done
"""

import os
import json
import logging
import re
import time
import math
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any
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
# Unique ID Generation
# =============================================================================

def make_unique_id(record: dict) -> str:
    """
    Generate a deterministic unique ID for a multi-function group record.

    Uses sample_id + group_type + sorted function names to ensure uniqueness
    and checkpoint consistency across restarts.
    """
    sample_id = record.get('sample_id', '')
    group_type = record.get('group_type', '')
    functions = record.get('functions', [])
    func_names = sorted(f.get('func_name', '') for f in functions)
    func_key = "|".join(func_names)
    return f"{sample_id}__{group_type}__{func_key}"


# =============================================================================
# Prompt Templates
# =============================================================================

# ── Original (non-guided) FIM prompt ──────────────────────────────────────────

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

# ── Guided FIM prompt (provides ground truth for higher-quality reasoning) ────

MULTI_FIM_COMPLETION_PROMPT_GUIDED = '''You are an expert Python programmer. Your task is to complete {num_functions} masked functions based on the surrounding code context.

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

## Critical Instructions for Reasoning
- In the Reasoning section, you must ONLY describe your analysis of the code context: the function signatures, how the functions are used, what patterns you observe, and what each function logically should do.
- Your reasoning must read as a natural, independent chain of thought — as if you are figuring out the implementations purely from the surrounding code.
- Do NOT mention, reference, or hint at any "reference", "hint", "example", "ground truth", or "provided implementation" in the Reasoning section. Write as if you are discovering the logic yourself.
- The Reasoning section should demonstrate genuine code comprehension: explain WHY each part of the implementation makes sense given the context.
- When reasoning about cross-function relationships, explain how you infer the contracts between functions from their usage patterns in the code — not from any reference.

## Critical Instructions for Implementation
- You are given reference implementations below to help you understand each function's PURPOSE, LOGIC, and EDGE CASES.
- However, you must write YOUR OWN version of each implementation. Do NOT copy the references line by line.
- Each implementation must be **functionally equivalent** (same inputs → same outputs, same side effects) but should reflect your own coding choices:
  - You may use different variable names
  - You may restructure the control flow (e.g., early returns vs. if-else chains, list comprehensions vs. loops)
  - You may reorder independent statements
  - You may use different but equivalent standard library calls
  - You may add or adjust comments in your own words
- The goal is: a competent programmer who understands the same requirements would write it this way — not a copy-paste of the reference.
- If a reference implementation is very short or trivial (e.g., a single return statement, a one-liner), it is acceptable for your implementation to look very similar, since there is essentially only one natural way to write it.
- Cross-function consistency is paramount: make sure your implementations work together correctly. If function A calls function B, your version of A must be compatible with your version of B.

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

## Reference Implementations (for understanding intent only — do NOT copy verbatim)

Study these implementations to understand what each function should accomplish, what edge cases they handle, and what APIs they use. Then write your own versions that achieve the same behavior while maintaining cross-function consistency.

{reference_sections}

Please analyze the context and provide your reasoning followed by the implementations.
'''


MULTI_CRITIQUE_PROMPT = '''You are an expert code reviewer. Your task is to evaluate a **group** of {num_functions} function completions against their ground truth implementations, AND determine whether each function and the group as a whole are suitable for FIM evaluation.

## Context

A model was asked to complete {num_functions} masked functions simultaneously. These functions are related: they {relationship_description}.

## Code with Masked Functions

```python
{masked_code}
```

## Per-Function Evaluation

{per_function_sections}

## Part 1: Per-Function Feasibility & Quality

For EACH function, evaluate:

**Feasibility**: Is the function completable from context alone?
- **Infeasible** if it relies on obscure external APIs, domain-specific conventions not in the file, or magic constants that cannot be guessed.
- **Feasible** if its behavior can be reasonably inferred from the function name, signature, docstring, surrounding code, and the OTHER functions in the group.

**Quality Scores** (1-5 scale):
1. **Correctness**: Does it produce the same results as ground truth?
2. **Executability**: Can it run without errors?
3. **API Usage**: Does it correctly use other functions/methods in the codebase?
4. **Completeness**: Does it handle all cases the ground truth handles?

## Part 2: Group Coherence

Evaluate the **cross-function consistency** of the completions:

1. **Interface Consistency (1-5)**: Do the functions correctly call each other? Do parameter types, return types, and calling conventions match between the completed functions?
2. **State Consistency (1-5)**: If functions share instance variables (self.xxx), are reads and writes consistent? Does one function write what the other reads?
3. **Logic Consistency (1-5)**: Do the functions implement a coherent overall design? Would they work correctly together as a unit?

## Output Format

```json
{{
  "per_function": [
    {{
      "func_name": "<function name>",
      "feasibility": {{
        "is_feasible": <true or false>,
        "confidence": <0.0 to 1.0>,
        "reason": "<explanation>"
      }},
      "scores": {{
        "correctness": <1-5>,
        "correctness_reason": "<brief explanation>",
        "executability": <1-5>,
        "executability_reason": "<brief explanation>",
        "api_usage": <1-5>,
        "api_usage_reason": "<brief explanation>",
        "completeness": <1-5>,
        "completeness_reason": "<brief explanation>"
      }},
      "overall_score": <1-5>,
      "overall_reason": "<summary>"
    }}
  ],
  "group_coherence": {{
    "interface_consistency": <1-5>,
    "interface_consistency_reason": "<explanation>",
    "state_consistency": <1-5>,
    "state_consistency_reason": "<explanation>",
    "logic_consistency": <1-5>,
    "logic_consistency_reason": "<explanation>",
    "coherence_score": <1-5, average of the three above>,
    "coherence_reason": "<overall coherence summary>"
  }},
  "group_overall_score": <1-5, considering both individual quality and coherence>,
  "group_overall_reason": "<summary>",
  "should_discard": <true or false>,
  "discard_reason": "<if should_discard is true, explain why>"
}}
```

## Decision Logic for `should_discard`

Set `should_discard` to `true` if ANY of the following:
1. **Any function is infeasible** — the group contains a function that fundamentally cannot be completed from context.
2. **Any function has executability ≤ 1** — a non-functional completion poisons the group.
3. **Any function has overall_score ≤ 1** — one very bad function makes the group unusable.
4. **Group coherence_score ≤ 2** — the functions don't work together coherently.
5. **group_overall_score ≤ 2** — the group as a whole is too poor.

Otherwise set `should_discard` to `false`.

Please provide your detailed analysis.
'''


# =============================================================================
# Relationship Descriptions for Prompts
# =============================================================================

GROUP_TYPE_DESCRIPTIONS = {
    # Pairs
    "caller_callee": "have a caller-callee relationship (one calls the other)",
    "co_callee": "are both called by the same function (co-callees)",
    "sibling_coupled": "are methods of the same class that share instance variables",
    "mutual_call": "call each other (mutual dependency)",
    # Triples
    "call_chain": "form a call chain (A calls B, B calls C)",
    "hub": "have a hub pattern (one function calls the other two)",
    "fan_in": "have a fan-in pattern (two functions both call the third)",
    "class_triad": "are three methods of the same class sharing instance variables",
}


def get_relationship_description(group_type: str) -> str:
    return GROUP_TYPE_DESCRIPTIONS.get(group_type, "are structurally related")


def build_function_output_sections(functions: List[Dict]) -> str:
    """Build the per-function output format instructions for the FIM prompt."""
    sections = []
    for i, fn in enumerate(functions, 1):
        name = fn.get('func_name', f'function_{i}')
        sections.append(
            f"### Function {i}: `{name}`\n"
            f"```python\n<The complete function body for {name}>\n```"
        )
    return "\n\n".join(sections)


def build_function_list(functions: List[Dict]) -> str:
    """Build the function list for the FIM prompt."""
    lines = []
    for i, fn in enumerate(functions, 1):
        name = fn.get('func_name', '')
        loc = fn.get('loc', '?')
        lines.append(f"{i}. `{name}` (approx. {loc} lines)")
    return "\n".join(lines)


def build_reference_sections(functions: List[Dict]) -> str:
    """Build reference implementation sections for the guided FIM prompt."""
    sections = []
    for i, fn in enumerate(functions, 1):
        name = fn.get('func_name', '')
        ground_truth = fn.get('func_content', '')
        sections.append(
            f"### Function {i}: `{name}`\n"
            f"```python\n{ground_truth}\n```"
        )
    return "\n\n".join(sections)


def build_per_function_critique_sections(functions: List[Dict], completions: Dict[str, str]) -> str:
    """Build per-function ground truth vs completion sections for the critique prompt."""
    sections = []
    for i, fn in enumerate(functions, 1):
        name = fn.get('func_name', '')
        ground_truth = fn.get('func_content', '')
        completion = completions.get(name, '<NO COMPLETION EXTRACTED>')
        sections.append(
            f"### Function {i}: `{name}`\n\n"
            f"**Ground Truth:**\n```python\n{ground_truth}\n```\n\n"
            f"**Model's Completion:**\n```python\n{completion}\n```"
        )
    return "\n\n".join(sections)


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
        return {
            'input_tokens': self.total_input_tokens,
            'output_tokens': self.total_output_tokens,
            'total_tokens': self.total_input_tokens + self.total_output_tokens,
            'input_cost': input_cost,
            'output_cost': output_cost,
            'total_cost': input_cost + output_cost,
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
        for key in [
            'total_input_tokens', 'total_output_tokens', 'request_count',
            'fim_input_tokens', 'fim_output_tokens', 'fim_requests',
            'critique_input_tokens', 'critique_output_tokens', 'critique_requests',
        ]:
            setattr(self, key, data.get(key, 0))


class GeminiClient:
    """Gemini API client for code completion and critique."""

    def __init__(self, model: str = "gemini-3-flash-preview",
                 token_counter: TokenCounter = None,
                 use_wandb: bool = False):
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
    ) -> tuple:
        last_error = None
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config={"temperature": temperature, "top_p": 0.95},
                )
                latency = time.time() - start_time
                usage_info = self._extract_usage(response, call_type)
                usage_info['latency'] = latency
                if self.use_wandb:
                    self._log_to_wandb(sample_id, call_type, usage_info, latency, temperature)
                return response.text, usage_info
            except Exception as e:
                last_error = e
                wait = retry_delay * (2 ** attempt)
                logger.warning(
                    f"Gemini API call failed (attempt {attempt + 1}/{max_retries}): {e}. "
                    f"Retrying in {wait:.1f}s..."
                )
                if attempt < max_retries - 1:
                    time.sleep(wait)

        logger.error(f"Gemini API call failed after {max_retries} attempts: {last_error}")
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

    def _log_to_wandb(self, sample_id, call_type, usage_info, latency, temperature):
        input_cost = (usage_info['input_tokens'] / 1_000_000) * PRICING['input']
        output_cost = (usage_info['output_tokens'] / 1_000_000) * PRICING['output']
        wandb.log({
            f"api_call/{call_type}_sample_id": sample_id,
            f"api_call/{call_type}_input_tokens": usage_info['input_tokens'],
            f"api_call/{call_type}_output_tokens": usage_info['output_tokens'],
            f"api_call/{call_type}_cost": input_cost + output_cost,
            f"api_call/{call_type}_latency": latency,
        })


# =============================================================================
# Response Parsing
# =============================================================================

def parse_multi_fim_response(response: str, function_names: List[str]) -> dict:
    """
    Parse a multi-function FIM response.

    Expected format has multiple code blocks, one per function, each preceded
    by a heading like "### Function 1: `func_name`".

    Fallback: if only one code block is found for multiple functions, tries
    splitting by function signature patterns.
    """
    result = {
        'raw_response': response,
        'reasoning': '',
        'implementations': {},  # func_name -> implementation string
        'parse_success': False,
    }
    if not response:
        return result

    # Extract reasoning
    reasoning_match = re.search(
        r'###?\s*Reasoning\s*\n(.*?)(?=###?\s*Function\s+\d|$)',
        response, re.DOTALL | re.IGNORECASE
    )
    if reasoning_match:
        result['reasoning'] = reasoning_match.group(1).strip()

    # Strategy 1: look for named function sections
    for fname in function_names:
        short_name = fname.split('.')[-1]
        # Match "### Function N: `ClassName.method`" or "### Function N: `method`"
        pattern = (
            rf'###?\s*Function\s+\d+\s*:\s*`?{re.escape(fname)}`?\s*\n'
            rf'.*?```python\s*\n(.*?)\n```'
        )
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if not match:
            # Try short name
            pattern = (
                rf'###?\s*Function\s+\d+\s*:\s*`?{re.escape(short_name)}`?\s*\n'
                rf'.*?```python\s*\n(.*?)\n```'
            )
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            result['implementations'][fname] = match.group(1).strip()

    # Strategy 2: if named sections didn't work, try ordered code blocks
    if len(result['implementations']) < len(function_names):
        code_blocks = re.findall(r'```python\s*\n(.*?)\n```', response, re.DOTALL)
        # Skip the first block if it looks like it might be the reasoning/context
        # (heuristic: if there are more blocks than functions, skip the first)
        if len(code_blocks) > len(function_names):
            code_blocks = code_blocks[-len(function_names):]

        if len(code_blocks) == len(function_names):
            for fname, block in zip(function_names, code_blocks):
                if fname not in result['implementations']:
                    result['implementations'][fname] = block.strip()

    # Check if we got all functions
    if len(result['implementations']) == len(function_names):
        result['parse_success'] = True
    elif len(result['implementations']) > 0:
        # Partial success — log warning but still usable
        logger.warning(
            f"Partial parse: got {len(result['implementations'])}/{len(function_names)} "
            f"functions. Missing: "
            f"{set(function_names) - set(result['implementations'].keys())}"
        )
        result['parse_success'] = True  # partial is still usable

    return result


def parse_multi_critique_response(response: str) -> dict:
    """Parse the multi-function critique response."""
    result = {
        'raw_response': response,
        'parse_success': False,
        'per_function': [],
        'group_coherence': {},
        'group_overall_score': 0,
        'group_overall_reason': '',
        'should_discard': False,
        'discard_reason': '',
    }
    if not response:
        return result

    # Extract JSON
    json_match = re.search(r'```json\s*\n(.*?)\n```', response, re.DOTALL)
    if json_match:
        json_str = json_match.group(1).strip()
    else:
        json_str = extract_json_object(response)

    if not json_str:
        logger.warning("Could not extract JSON from multi-critique response")
        return result

    try:
        parsed = json.loads(json_str)
        result['per_function'] = parsed.get('per_function', [])
        result['group_coherence'] = parsed.get('group_coherence', {})
        result['group_overall_score'] = parsed.get('group_overall_score', 0)
        result['group_overall_reason'] = parsed.get('group_overall_reason', '')
        result['should_discard'] = parsed.get('should_discard', False)
        result['discard_reason'] = parsed.get('discard_reason', '')
        result['parse_success'] = True
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse multi-critique JSON: {e}")

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
# Pre-Sharding
# =============================================================================

def pre_shard_input(input_path: str, total_shards: int):
    """
    Split the large input JSON into per-shard files on disk.

    Creates files like: input_shard_1_of_50.json, input_shard_2_of_50.json, ...
    Each shard file contains only the records for that shard.

    This is run ONCE before launching parallel workers, so each worker
    only reads its own small shard file (avoids OOM).
    """
    input_p = Path(input_path)
    logger.info(f"📂 Pre-sharding: loading {input_p}")
    with open(input_p, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total = len(data)
    logger.info(f"   Total records: {total}, splitting into {total_shards} shards")

    shard_dir = input_p.parent / f"{input_p.stem}_shards_{total_shards}"
    shard_dir.mkdir(exist_ok=True)

    for shard_id in range(1, total_shards + 1):
        base_size = total // total_shards
        remainder = total % total_shards
        if shard_id <= remainder:
            start = (shard_id - 1) * (base_size + 1)
            end = start + base_size + 1
        else:
            start = remainder * (base_size + 1) + (shard_id - 1 - remainder) * base_size
            end = start + base_size

        shard_data = data[start:end]
        shard_path = shard_dir / f"shard_{shard_id}_of_{total_shards}.json"
        with open(shard_path, 'w', encoding='utf-8') as f:
            json.dump(shard_data, f, ensure_ascii=False)
        logger.info(f"   Shard {shard_id}: [{start}, {end}) → {len(shard_data)} records → {shard_path}")

    logger.info(f"✅ Pre-sharding complete. Shard directory: {shard_dir}")
    return str(shard_dir)


def get_shard_file_path(input_path: str, shard_id: int, total_shards: int) -> Path:
    """Get the path to a pre-sharded file."""
    input_p = Path(input_path)
    shard_dir = input_p.parent / f"{input_p.stem}_shards_{total_shards}"
    return shard_dir / f"shard_{shard_id}_of_{total_shards}.json"


# =============================================================================
# Main Pipeline
# =============================================================================

class MultiFIMCompletionPipeline:
    """Pipeline for multi-function FIM completion and critique."""

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
        wandb_project: str = "multi-fim-completion-critique",
        wandb_run_name: str = None,
        shard_id: int = None,
        total_shards: int = None,
        start_idx: int = None,
        end_idx: int = None,
        guided: bool = False,
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

        # Sharding
        self.shard_id = shard_id
        self.total_shards = total_shards
        self.start_idx = start_idx
        self.end_idx = end_idx

        # Token counter
        self.token_counter = TokenCounter()

        # W&B
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name

        # Adjust paths for sharding
        self._adjust_paths_for_sharding()

        # Init wandb
        if self.use_wandb:
            self._init_wandb()

        # Init Gemini
        self.gemini_client = GeminiClient(
            model=model,
            token_counter=self.token_counter,
            use_wandb=self.use_wandb
        )

        # Checkpoint
        self.processed_ids = set()
        self.results_pair = []      # group_size == 2
        self.results_triple = []    # group_size == 3
        self._load_checkpoint()

        # Log mode
        mode_str = "GUIDED (with ground truth)" if self.guided else "STANDARD (no ground truth)"
        logger.info(f"🔧 FIM mode: {mode_str}")

    # -----------------------------------------------------------------
    # Path management
    # -----------------------------------------------------------------

    def _adjust_paths_for_sharding(self):
        shard_suffix = None
        if self.shard_id is not None and self.total_shards is not None:
            shard_suffix = f"_shard{self.shard_id}_of_{self.total_shards}"
        elif self.start_idx is not None or self.end_idx is not None:
            start = self.start_idx if self.start_idx is not None else 0
            end = self.end_idx if self.end_idx is not None else "end"
            shard_suffix = f"_range_{start}_{end}"

        if shard_suffix:
            for attr in ['output_path', 'checkpoint_path']:
                p = getattr(self, attr)
                new_p = p.parent / f"{p.stem}{shard_suffix}{p.suffix}"
                setattr(self, attr, new_p)
            logger.info(f"📂 Adjusted output: {self.output_path}")
            logger.info(f"   Adjusted checkpoint: {self.checkpoint_path}")

    def _get_output_path(self, group_size: int) -> Path:
        """Get separate output paths for pair vs triple results."""
        stem = self.output_path.stem
        suffix = self.output_path.suffix
        tag = "pairs" if group_size == 2 else "triples"
        return self.output_path.parent / f"{stem}_{tag}{suffix}"

    # -----------------------------------------------------------------
    # Sharding
    # -----------------------------------------------------------------

    def _load_shard_data(self) -> List[Dict]:
        """
        Load data for this shard. Uses pre-sharded files if available,
        otherwise falls back to loading full file + slicing.
        """
        if self.shard_id is not None and self.total_shards is not None:
            shard_file = get_shard_file_path(
                str(self.input_path), self.shard_id, self.total_shards
            )
            if shard_file.exists():
                logger.info(f"📂 Loading pre-sharded file: {shard_file}")
                with open(shard_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
                # No need to slice — file already contains only this shard's data
            else:
                logger.warning(
                    f"Pre-shard file not found: {shard_file}. "
                    f"Falling back to full-file loading. "
                    f"Run --pre-shard first to avoid OOM."
                )

        # Fallback: load full file
        logger.info(f"📂 Loading full input: {self.input_path}")
        with open(self.input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        total = len(data)
        start, end = self._get_shard_range(total)
        logger.info(f"   Slicing [{start}, {end}) out of {total}")
        return data[start:end]

    def _get_shard_range(self, total: int) -> tuple:
        if self.start_idx is not None or self.end_idx is not None:
            start = max(0, min(self.start_idx or 0, total))
            end = max(start, min(self.end_idx or total, total))
            return start, end
        if self.shard_id is not None and self.total_shards is not None:
            base = total // self.total_shards
            rem = total % self.total_shards
            if self.shard_id <= rem:
                start = (self.shard_id - 1) * (base + 1)
                end = start + base + 1
            else:
                start = rem * (base + 1) + (self.shard_id - 1 - rem) * base
                end = start + base
            return start, end
        return 0, total

    # -----------------------------------------------------------------
    # W&B
    # -----------------------------------------------------------------

    def _init_wandb(self):
        base_name = self.wandb_run_name or f"multi-fim-{time.strftime('%Y%m%d-%H%M%S')}"
        if self.shard_id is not None:
            run_name = f"{base_name}-shard{self.shard_id}"
        elif self.start_idx is not None:
            run_name = f"{base_name}-range{self.start_idx}-{self.end_idx or 'end'}"
        else:
            run_name = base_name
        wandb.init(
            project=self.wandb_project, name=run_name,
            config={
                "model": self.model,
                "input_path": str(self.input_path),
                "shard_id": self.shard_id,
                "total_shards": self.total_shards,
                "guided": self.guided,
            },
            resume="allow"
        )
        weave.init(self.wandb_project)
        logger.info(f"✅ W&B initialized: {wandb.run.url}")

    # -----------------------------------------------------------------
    # Checkpoint
    # -----------------------------------------------------------------

    def _load_checkpoint(self):
        if not self.checkpoint_path.exists():
            return
        try:
            with open(self.checkpoint_path, 'r', encoding='utf-8') as f:
                ckpt = json.load(f)
            self.processed_ids = set(ckpt.get('processed_ids', []))
            self.results_pair = ckpt.get('results_pair', [])
            self.results_triple = ckpt.get('results_triple', [])
            if 'token_usage' in ckpt:
                self.token_counter.load_from_dict(ckpt['token_usage'])
            logger.info(
                f"📂 Checkpoint loaded: {len(self.processed_ids)} processed, "
                f"{len(self.results_pair)} pairs, {len(self.results_triple)} triples"
            )
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")

    def _save_checkpoint(self):
        ckpt = {
            'processed_ids': list(self.processed_ids),
            'results_pair': self.results_pair,
            'results_triple': self.results_triple,
            'token_usage': self.token_counter.to_dict(),
        }
        temp = self.checkpoint_path.with_suffix('.tmp')
        try:
            with open(temp, 'w', encoding='utf-8') as f:
                json.dump(ckpt, f, ensure_ascii=False, indent=2)
            temp.replace(self.checkpoint_path)
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            if temp.exists():
                temp.unlink()

    # -----------------------------------------------------------------
    # Debug helpers
    # -----------------------------------------------------------------

    def _print_response(self, title: str, uid: str, response: str, usage: dict):
        print("\n" + "=" * 80)
        print(f"📝 {title} for: {uid}")
        print(f"   Tokens: in={usage['input_tokens']:,}, out={usage['output_tokens']:,}, "
              f"latency={usage.get('latency', 0):.2f}s")
        print("=" * 80)
        print(response[:3000] + "..." if len(response) > 3000 else response)
        print("=" * 80 + "\n")

    def _print_current_cost(self):
        cost = self.token_counter.get_cost()
        print(f"💰 Cumulative: ${cost['total_cost']:.4f} "
              f"(in:{cost['input_tokens']:,} out:{cost['output_tokens']:,} "
              f"FIM:{cost['fim_requests']} Critique:{cost['critique_requests']})")

    # -----------------------------------------------------------------
    # Core: process one group record
    # -----------------------------------------------------------------

    def process_single_record(self, record: dict) -> Optional[dict]:
        """
        Process one group record: multi-FIM completion + multi-critique.
        """
        unique_id = make_unique_id(record)
        functions = record.get('functions', [])
        function_names = [f['func_name'] for f in functions]
        masked_code = record.get('masked_code_content', '')
        group_type = record.get('group_type', '')
        group_size = record.get('group_size', len(functions))

        # Copy all original fields
        result = record.copy()
        result['unique_id'] = unique_id
        result['guided_mode'] = self.guided

        relationship_desc = get_relationship_description(group_type)

        # =================================================================
        # Step 1: Multi-FIM Completion
        # =================================================================
        if self.guided:
            # ── Guided mode: include ground truth references ──
            fim_prompt = MULTI_FIM_COMPLETION_PROMPT_GUIDED.format(
                num_functions=group_size,
                relationship_description=relationship_desc,
                function_output_sections=build_function_output_sections(functions),
                masked_code=masked_code,
                function_list=build_function_list(functions),
                reference_sections=build_reference_sections(functions),
            )
        else:
            # ── Standard mode: no ground truth ──
            fim_prompt = MULTI_FIM_COMPLETION_PROMPT.format(
                num_functions=group_size,
                relationship_description=relationship_desc,
                function_output_sections=build_function_output_sections(functions),
                masked_code=masked_code,
                function_list=build_function_list(functions),
            )

        try:
            fim_response, fim_usage = self.gemini_client.get_response(
                prompt=fim_prompt,
                sample_id=unique_id,
                call_type="fim",
                temperature=self.fim_temperature,
            )
            if self.print_response:
                self._print_response("Multi-FIM Completion", unique_id, fim_response, fim_usage)

            fim_parsed = parse_multi_fim_response(fim_response, function_names)

            result['fim_response'] = {
                'raw_response': fim_response,
                'reasoning': fim_parsed['reasoning'],
                'implementations': fim_parsed['implementations'],
                'parse_success': fim_parsed['parse_success'],
                'token_usage': fim_usage,
            }

            if self.wait_seconds > 0:
                time.sleep(self.wait_seconds)

        except Exception as e:
            logger.error(f"Multi-FIM failed for {unique_id}: {e}")
            result['fim_response'] = {'error': str(e), 'parse_success': False}
            result['should_discard'] = None
            result['discard_reason'] = f"FIM API call failed: {e}"
            return result

        # =================================================================
        # Step 2: Multi-Critique
        # =================================================================
        completions = fim_parsed['implementations'] if fim_parsed['parse_success'] else {}

        critique_prompt = MULTI_CRITIQUE_PROMPT.format(
            num_functions=group_size,
            relationship_description=relationship_desc,
            masked_code=masked_code,
            per_function_sections=build_per_function_critique_sections(functions, completions),
        )

        try:
            critique_response, critique_usage = self.gemini_client.get_response(
                prompt=critique_prompt,
                sample_id=unique_id,
                call_type="critique",
                temperature=self.critique_temperature,
            )
            if self.print_response:
                self._print_response("Multi-Critique", unique_id, critique_response, critique_usage)

            critique_parsed = parse_multi_critique_response(critique_response)

            result['critique_response'] = {
                'raw_response': critique_response,
                'per_function': critique_parsed['per_function'],
                'group_coherence': critique_parsed['group_coherence'],
                'group_overall_score': critique_parsed['group_overall_score'],
                'group_overall_reason': critique_parsed['group_overall_reason'],
                'should_discard': critique_parsed['should_discard'],
                'discard_reason': critique_parsed['discard_reason'],
                'parse_success': critique_parsed['parse_success'],
                'token_usage': critique_usage,
            }

            # Promote to top-level
            result['should_discard'] = critique_parsed['should_discard']
            result['discard_reason'] = critique_parsed['discard_reason']
            result['group_overall_score'] = critique_parsed['group_overall_score']
            result['group_coherence_score'] = critique_parsed.get(
                'group_coherence', {}
            ).get('coherence_score', 0)

            # Compute per-function feasibility summary
            per_fn = critique_parsed.get('per_function', [])
            result['all_feasible'] = all(
                f.get('feasibility', {}).get('is_feasible', True)
                for f in per_fn
            ) if per_fn else None

            if self.wait_seconds > 0:
                time.sleep(self.wait_seconds)

        except Exception as e:
            logger.error(f"Multi-critique failed for {unique_id}: {e}")
            result['critique_response'] = {'error': str(e), 'parse_success': False}
            result['should_discard'] = None
            result['discard_reason'] = f"Critique API call failed: {e}"

        # W&B logging
        if self.use_wandb and result.get('critique_response', {}).get('parse_success'):
            cr = result['critique_response']
            log_data = {
                "scores/group_overall": cr['group_overall_score'],
                "coherence/score": cr.get('group_coherence', {}).get('coherence_score', 0),
                "decision/should_discard": cr['should_discard'],
                "meta/group_size": group_size,
                "meta/group_type": group_type,
                "meta/guided": self.guided,
            }
            for pf in cr.get('per_function', []):
                fn = pf.get('func_name', 'unknown')
                log_data[f"per_func/{fn}/overall"] = pf.get('overall_score', 0)
            wandb.log(log_data)

        return result

    # -----------------------------------------------------------------
    # Run pipeline
    # -----------------------------------------------------------------

    def run(self):
        """Run the multi-function FIM pipeline."""
        data = self._load_shard_data()
        logger.info(f"📊 Records in this shard: {len(data)}")

        # Assign unique IDs
        for record in data:
            record['_unique_id'] = make_unique_id(record)

        remaining = [r for r in data if r['_unique_id'] not in self.processed_ids]
        logger.info(f"   Already processed: {len(self.processed_ids)}")
        logger.info(f"   Remaining: {len(remaining)}")
        self._print_current_cost()

        shard_prefix = f"[Shard {self.shard_id}/{self.total_shards}] " if self.shard_id else ""
        mode_tag = "[guided] " if self.guided else ""

        with tqdm(total=len(remaining), desc=f"{shard_prefix}{mode_tag}Processing") as pbar:
            for record in remaining:
                uid = record['_unique_id']
                cost = self.token_counter.get_cost()
                pbar.set_description(
                    f"{shard_prefix}{mode_tag}Done:{len(self.processed_ids)} "
                    f"Cost:${cost['total_cost']:.3f}"
                )

                try:
                    record_clean = {k: v for k, v in record.items() if k != '_unique_id'}
                    result = self.process_single_record(record_clean)

                    if result:
                        gs = result.get('group_size', 2)
                        if gs == 3:
                            self.results_triple.append(result)
                        else:
                            self.results_pair.append(result)
                        self.processed_ids.add(uid)

                        # Log status
                        score = result.get('group_overall_score', 'N/A')
                        coh = result.get('group_coherence_score', 'N/A')
                        disc = result.get('should_discard', 'N/A')
                        logger.info(
                            f"  ✅ {uid}: score={score}, coherence={coh}, discard={disc}"
                        )

                except Exception as e:
                    logger.error(f"Error processing {uid}: {e}")
                    self.processed_ids.add(uid)

                self._save_checkpoint()
                self._print_current_cost()
                pbar.update(1)

        # Save final outputs (separate files for pairs and triples)
        for group_size, results, tag in [
            (2, self.results_pair, "pairs"),
            (3, self.results_triple, "triples"),
        ]:
            if results:
                out_path = self._get_output_path(group_size)
                logger.info(f"💾 Saving {len(results)} {tag} → {out_path}")
                with open(out_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)

        self._print_summary()

        if self.use_wandb:
            wandb.finish()

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------

    def _print_summary(self):
        all_results = self.results_pair + self.results_triple
        print("\n" + "=" * 60)
        print("📊 Multi-Function FIM Processing Summary")
        if self.shard_id:
            print(f"   Shard: {self.shard_id}/{self.total_shards}")
        print(f"   Mode: {'GUIDED' if self.guided else 'STANDARD'}")
        print("=" * 60)

        print(f"\n📈 Counts:")
        print(f"   Total processed: {len(self.processed_ids)}")
        print(f"   Pair results:    {len(self.results_pair)}")
        print(f"   Triple results:  {len(self.results_triple)}")

        self.token_counter.print_cost_summary()

        if all_results:
            # Discard stats
            keep = sum(1 for r in all_results if r.get('should_discard') is False)
            discard = sum(1 for r in all_results if r.get('should_discard') is True)
            unknown = sum(1 for r in all_results if r.get('should_discard') is None)
            print(f"\n🗑️ Discard Decision:")
            print(f"   Keep: {keep}  Discard: {discard}  Unknown: {unknown}")

            # Group type breakdown
            type_counts = {}
            for r in all_results:
                gt = r.get('group_type', '?')
                type_counts[gt] = type_counts.get(gt, 0) + 1
            print(f"\n📋 Group Type Breakdown:")
            for gt, cnt in sorted(type_counts.items(), key=lambda x: -x[1]):
                print(f"   {gt}: {cnt}")

            # Score distribution
            scores = [
                r.get('group_overall_score', 0)
                for r in all_results
                if r.get('critique_response', {}).get('parse_success')
            ]
            if scores:
                avg = sum(scores) / len(scores)
                print(f"\n📊 Group Overall Score:")
                print(f"   Average: {avg:.2f}")
                dist = {}
                for s in scores:
                    dist[s] = dist.get(s, 0) + 1
                for s in sorted(dist):
                    print(f"   Score {s}: {dist[s]} ({dist[s]/len(scores)*100:.1f}%)")

            # Coherence scores
            coh_scores = [
                r.get('group_coherence_score', 0)
                for r in all_results
                if r.get('group_coherence_score') is not None
                and r.get('group_coherence_score') > 0
            ]
            if coh_scores:
                print(f"\n🔗 Group Coherence Score:")
                print(f"   Average: {sum(coh_scores)/len(coh_scores):.2f}")

            # Parse success
            fim_ok = sum(1 for r in all_results if r.get('fim_response', {}).get('parse_success'))
            cri_ok = sum(1 for r in all_results if r.get('critique_response', {}).get('parse_success'))
            n = len(all_results)
            print(f"\n📋 Parse Success Rate:")
            print(f"   FIM: {fim_ok}/{n} ({fim_ok/max(n,1)*100:.1f}%)")
            print(f"   Critique: {cri_ok}/{n} ({cri_ok/max(n,1)*100:.1f}%)")

        print("\n" + "=" * 60)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Multi-Function FIM Completion & Critique Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Step 0: pre-shard (run once)
  python step_4_multi_fim_gemini.py --pre-shard \\
    -i .../step_3_selected_multi_fim_functions_0227_groups.json \\
    --total-shards 50

  # Step 1: run each shard in parallel (guided mode)
  for i in $(seq 1 50); do
    nohup python step_4_multi_fim_gemini.py \\
      -i .../step_3_selected_multi_fim_functions_0227_groups.json \\
      -o .../step_4_multi_fim_output_0227.json \\
      --shard $i --total-shards 50 --guided \\
      > shard_$i.log 2>&1 &
  done

  # Or single process (standard mode):
  python step_4_multi_fim_gemini.py \\
    -i .../groups.json -o .../output.json

  # Single process (guided mode):
  python step_4_multi_fim_gemini.py \\
    -i .../groups.json -o .../output.json --guided
        '''
    )

    # Special mode: pre-shard only
    parser.add_argument("--pre-shard", action="store_true",
                        help="Pre-split input into per-shard files, then exit")

    # I/O
    parser.add_argument("--input", "-i", required=True,
                        help="Path to groups JSON (from depfim_multi.py)")
    parser.add_argument("--output", "-o", default=None,
                        help="Path to output JSON (not needed for --pre-shard)")
    parser.add_argument("--checkpoint", "-c", default=None,
                        help="Checkpoint file path (auto-generated if omitted)")

    # Model
    parser.add_argument("--model", "-m", default="gemini-3-flash-preview")
    parser.add_argument("--fim-temperature", type=float, default=0.7)
    parser.add_argument("--critique-temperature", type=float, default=0.3)

    # Processing
    parser.add_argument("--wait", "-w", type=float, default=0.5)
    parser.add_argument("--print-response", "-p", action="store_true")

    # Guided mode
    parser.add_argument("--guided", action="store_true",
                        help="Enable guided mode: provide ground truth to the model "
                             "for higher-quality reasoning (reasoning will still read "
                             "as independent analysis)")

    # Sharding
    parser.add_argument("--shard", type=int, default=None, help="Shard ID (1-indexed)")
    parser.add_argument("--total-shards", type=int, default=None)
    parser.add_argument("--start-idx", type=int, default=None)
    parser.add_argument("--end-idx", type=int, default=None)

    # W&B
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="multi-fim-completion-critique")
    parser.add_argument("--wandb-run-name", default=None)

    args = parser.parse_args()

    # --pre-shard mode
    if args.pre_shard:
        if args.total_shards is None:
            parser.error("--pre-shard requires --total-shards")
        pre_shard_input(args.input, args.total_shards)
        return

    # Normal mode
    if args.output is None:
        parser.error("-o / --output is required (unless using --pre-shard)")

    if args.checkpoint is None:
        out_p = Path(args.output)
        args.checkpoint = str(out_p.parent / f"{out_p.stem}_checkpoint.json")

    # Validate sharding args
    if args.shard is not None and args.total_shards is None:
        parser.error("--shard requires --total-shards")
    if args.total_shards is not None and args.shard is None and not args.pre_shard:
        parser.error("--total-shards requires --shard (or use --pre-shard)")
    if args.shard is not None and (args.start_idx is not None or args.end_idx is not None):
        parser.error("Cannot use --shard with --start-idx / --end-idx")

    if args.wandb and not WANDB_AVAILABLE:
        logger.warning("⚠️ W&B requested but not available")
        args.wandb = False

    pipeline = MultiFIMCompletionPipeline(
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
        guided=args.guided,
    )
    pipeline.run()


if __name__ == "__main__":
    main()