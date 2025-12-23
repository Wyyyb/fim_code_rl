
select_function_prompt_ch = """你是一个代码分析专家，现在需要帮助我完成一个 Fill-in-the-Middle (FIM) 训练数据集的构建任务。

## 任务背景

Fill-in-the-Middle 是一种代码补全训练任务。我们的目标是：从一个完整的 Python 文件中选择合适的函数"扣掉"，让模型根据上下文（前缀代码 + 后缀代码）来补全被扣掉的函数体。理想情况下，被选中的函数应该：
- 与文件中的其他代码存在调用关系（例如：函数 A 调用函数 B，函数 B 调用函数 C，那么扣掉函数 B 是一个好的选择，因为上下文提供了足够的信息来推断 B 的功能）
- 具有适中的补全难度，既不能过于简单（如简单的 getter/setter），也不能因为上下文信息不足而无法合理推断

## 你的任务

请对以下 Python 代码进行分析，完成两个步骤：

### 步骤一：代码质量评估

请从以下三个维度评估这份代码是否适合用于 FIM 任务，每个维度给出 1-5 分的评分（5 分最高）：

1. **复杂度 (Complexity)**：代码是否具有足够的逻辑复杂度？过于简单的代码（如仅包含几个独立的工具函数）不适合此任务。
2. **代码质量 (Quality)**：代码是否语法正确、功能完整、风格规范？存在明显错误或不完整的代码不适合。
3. **内聚性 (Cohesion)**：代码中的函数之间是否存在调用关系或逻辑关联？如果所有函数都是完全独立、互不相关的工具函数集合，则不适合此任务。

根据三个维度的评分，给出最终判定：**适合** 或 **不适合** 用于 FIM 任务。
- 判定标准建议：任一维度低于 2 分，或平均分低于 3 分，则判定为"不适合"

### 步骤二：选择适合扣掉的函数

如果步骤一判定为"适合"，请选择适合作为 FIM 补全目标的函数。选择标准：

1. **长度要求**：函数体大约在 10-100 行之间（过短缺乏训练价值，过长增加补全难度）
2. **上下文充分性**：扣掉该函数后，剩余代码（包括调用该函数的地方、被该函数调用的其他函数、相关注释和类型提示等）应该提供足够的信息来推断该函数的功能和实现
3. **难度适中**：既不能是简单的样板代码（如简单赋值、直接返回），也不能需要领域专业知识或外部信息才能实现

对于每个选中的函数，请提供：
- 函数名
- 补全难度评分 (1-5 分，3 分为适中)
- 选择理由（简要说明为什么这个函数适合作为 FIM 任务的目标）

## 输出格式

请严格按照以下 JSON 格式输出：

{
  "code_evaluation": {
    "complexity_score": <1-5>,
    "complexity_reason": "<简要说明>",
    "quality_score": <1-5>,
    "quality_reason": "<简要说明>",
    "cohesion_score": <1-5>,
    "cohesion_reason": "<简要说明>",
    "average_score": <计算平均分>,
    "is_suitable": <true/false>,
    "rejection_reason": "<如果不适合，说明主要原因；如果适合则为 null>"
  },
  "selected_functions": [
    {
      "function_name": "<函数名>",
      "difficulty_score": <1-5>,
      "reason": "<选择该函数的理由>"
    }
  ]
}

注意：
- 如果 is_suitable 为 false，则 selected_functions 应为空数组 []
- 可以选择多个函数，每个函数将生成一条独立的训练数据
- 如果代码适合但没有符合条件的函数，selected_functions 也可以为空数组

## 待分析的代码

```python
"""

select_function_prompt = """# Fill-in-the-Middle (FIM) Dataset Construction

## Task Background
Fill-in-the-Middle is a code completion training task. Our goal is to select appropriate functions from a complete Python file to "mask out," and then train a model to complete the masked function body based on the surrounding context (prefix code + suffix code). Ideally, the selected function should:
* Have calling relationships with other code in the file (e.g., if function A calls function B, and function B calls function C, then masking function B is a good choice because the context provides sufficient information to infer B's functionality)
* Have moderate completion difficulty—neither too trivial (like simple getters/setters) nor impossible to reasonably infer due to insufficient context

## Your Task
You are a code analysis expert. Please analyze the following Python code and complete two steps:

### Step 1: Code Quality Evaluation
Evaluate whether this code is suitable for the FIM task based on the following three dimensions. Provide a score from 1-5 for each (5 being the highest):
1. Complexity: Does the code have sufficient logical complexity? Overly simple code (e.g., containing only a few independent utility functions) is not suitable.
2. Quality: Is the code syntactically correct, functionally complete, and well-styled? Code with obvious errors or incomplete implementations is not suitable.
3. Cohesion: Do the functions in the code have calling relationships or logical connections with each other? If all functions are completely independent and unrelated utility functions, it is not suitable for this task.

Based on the scores from all three dimensions, provide a final verdict: Suitable or Not Suitable for the FIM task.
* Suggested criteria: If any dimension scores below 2, or if the average score is below 3, the verdict should be "Not Suitable"

### Step 2: Select Functions to Mask
If Step 1 verdict is "Suitable," please select functions that are appropriate as FIM completion targets. Selection criteria:
1. Length requirement: The function body should be approximately 10-100 lines (too short lacks training value; too long increases completion difficulty)
2. Context sufficiency: After masking the function, the remaining code (including call sites, functions called by the target, related comments, type hints, etc.) should provide enough information to infer the function's purpose and implementation
3. Moderate difficulty: Should not be simple boilerplate code (like simple assignments or direct returns), nor should it require domain expertise or external information to implement

For each selected function, provide:
* Function name
* Difficulty score (1-5, where 3 is moderate)
* Reason for selection (briefly explain why this function is suitable as a FIM task target)

## Output Format
Please output strictly in the following JSON format:
```json
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
* If is_suitable is false, then selected_functions should be an empty array []
* Multiple functions can be selected; each will generate an independent training sample
* If the code is suitable but no functions meet the criteria, selected_functions can also be an empty array

## Code to Analyze

```python
"""

gemini_prompt_map = {
    "select_function_prompt": select_function_prompt,
    "select_function_prompt_ch": select_function_prompt_ch
}









