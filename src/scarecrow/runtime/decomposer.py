# src/scarecrow/runtime/decomposer.py

"""Task Decomposer.

用于判断用户输入是否包含多个可顺序执行的独立任务。

注意：
- Decomposer 不选择工具
- Decomposer 不输出 capability
- Decomposer 不执行任务
- Decomposer 只负责把明显多任务拆成自然语言子任务
"""

import json
import re

from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, SystemMessage

from scarecrow.config import LLMConfig
from scarecrow.llm import load_chat_model_from_config


class DecomposedTask(BaseModel):
    """一个可交给 Router 的自然语言子任务。"""

    text: str = Field(..., description="可直接交给 Router 的单个自然语言任务")
    depends_on_previous: bool = Field(
        default=False,
        description="该任务是否依赖前面任务产生的上下文或结果",
    )
    reason: str = Field(default="", description="为什么这是一个独立子任务")


class TaskDecomposition(BaseModel):
    """多任务拆解结果。"""

    is_multi_task: bool = False
    tasks: list[DecomposedTask] = Field(default_factory=list)


_DECOMPOSER_SYSTEM_PROMPT = """你是 Scarecrow 的 Task Decomposer。

你的职责：
- 判断用户输入是否包含多个可顺序执行的独立任务
- 如果是多任务，把它拆成若干自然语言子任务
- 如果是单任务，不要过度拆分
- 不选择工具
- 不输出 capability
- 不写代码
- 不执行任务
- 输出严格 JSON
- 不输出 markdown
- 不输出 ```json 代码块

你必须输出如下 JSON 结构：

{
  "is_multi_task": true,
  "tasks": [
    {
      "text": "第一个可独立执行的自然语言任务",
      "depends_on_previous": false,
      "reason": "简短原因"
    },
    {
      "text": "第二个可独立执行的自然语言任务",
      "depends_on_previous": true,
      "reason": "简短原因"
    }
  ]
}

拆分原则：

1. 不要过度拆分
以下属于一个单任务，不要拆：
- 系统探索某个数据文件，包括字段类型、缺失值、异常值
- 分析一份数据并给建议
- 检查缺失值和异常值
- 对某个数据文件做 EDA
- 解释一段报错并给修复建议

2. 应该拆分
以下情况通常是多任务：
- 用户多行输入，且每行是一个独立目标
- 用户明确要求先做 A，再做 B，然后做 C
- 一个请求中混合了列文件、预览文件、分析文件、生成方案等多个阶段
- 一个请求中包含多个相互独立的数据文件操作

3. 子任务要求
- 每个子任务必须是自然语言
- 每个子任务应该可以单独交给 Router
- 不要在子任务中写工具名
- 不要在子任务中写 capability
- 不要补充用户没有说的任务
- 保留用户原始语言
- 保留必要的文件名、路径片段、分析目标
- 删除纯控制命令，例如 /state、/reset、/help，除非用户明确要求解释这些命令

4. depends_on_previous
- 如果一个子任务依赖前面产生的数据状态、分析结果或上下文，设为 true
- 如果可以独立执行，设为 false

5. 单任务输出
如果不是多任务，输出：
{
  "is_multi_task": false,
  "tasks": [
    {
      "text": "原始用户请求",
      "depends_on_previous": false,
      "reason": "这是一个单一目标"
    }
  ]
}
"""


def should_call_decomposer(user_input: str) -> bool:
    """轻量 precheck：只判断是否值得调用 LLM Decomposer。

    这是成本门控，不负责最终拆分。
    真正拆分由 LLM Decomposer 决定。
    """

    text = user_input.strip()

    if not text:
        return False

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) >= 2:
        return True

    question_count = text.count("?") + text.count("？")
    if question_count >= 2:
        return True

    sequence_markers = [
        "然后",
        "接着",
        "再",
        "最后",
        "首先",
        "其次",
        "第一",
        "第二",
        "第三",
        "step",
        "Step",
    ]

    marker_hits = sum(1 for marker in sequence_markers if marker in text)
    return marker_hits >= 2


def decompose_user_input(
    user_input: str,
    cfg: LLMConfig,
    task_state_brief: str | None = None,
) -> TaskDecomposition:
    """判断并拆分用户输入。

    如果 precheck 认为明显是单任务，直接返回单任务结果，不调用 LLM。
    """

    cleaned = user_input.strip()

    if not cleaned:
        return TaskDecomposition(is_multi_task=False, tasks=[])

    if not should_call_decomposer(cleaned):
        return TaskDecomposition(
            is_multi_task=False,
            tasks=[
                DecomposedTask(
                    text=cleaned,
                    depends_on_previous=False,
                    reason="输入看起来是一个单一任务。",
                )
            ],
        )

    model = load_chat_model_from_config(cfg)

    state_block = task_state_brief or "暂无结构化任务状态。"

    user_prompt = f"""当前任务状态：

{state_block}

用户输入：

{cleaned}

请判断这是否是多任务。如果是，拆成可顺序执行的自然语言子任务。
输出严格 JSON。
"""

    try:
        msg = model.invoke(
            [
                SystemMessage(content=_DECOMPOSER_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]
        )
        content = _message_content_to_text(msg.content)
        data = _extract_json_object(content)
        result = TaskDecomposition.model_validate(data)
    except Exception:
        return _fallback_single_task(cleaned)

    return _normalize_decomposition(result, cleaned)


def _normalize_decomposition(
    result: TaskDecomposition,
    original_input: str,
) -> TaskDecomposition:
    """清理 LLM 输出，避免空任务或过度拆分。"""

    tasks: list[DecomposedTask] = []

    for task in result.tasks:
        text = task.text.strip()

        if not text:
            continue

        if text.startswith("/"):
            continue

        tasks.append(
            DecomposedTask(
                text=text,
                depends_on_previous=task.depends_on_previous,
                reason=task.reason.strip(),
            )
        )

    if not tasks:
        return _fallback_single_task(original_input)

    if len(tasks) == 1:
        return TaskDecomposition(is_multi_task=False, tasks=tasks)

    return TaskDecomposition(is_multi_task=True, tasks=tasks)


def _fallback_single_task(user_input: str) -> TaskDecomposition:
    return TaskDecomposition(
        is_multi_task=False,
        tasks=[
            DecomposedTask(
                text=user_input,
                depends_on_previous=False,
                reason="Decomposer 解析失败，按单任务保守处理。",
            )
        ],
    )


def _message_content_to_text(content) -> str:
    """把 LangChain message.content 统一转成字符串。"""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []

        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text:
                    parts.append(str(text))
            else:
                parts.append(str(item))

        return "\n".join(parts)

    return str(content)


def _extract_json_object(text: str) -> dict:
    """从模型输出中提取 JSON object。"""

    cleaned = text.strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(
            r"^```(?:json)?",
            "",
            cleaned.strip(),
            flags=re.IGNORECASE,
        ).strip()
        cleaned = re.sub(r"```$", "", cleaned.strip()).strip()

    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in decomposer output: {text}")

    data = json.loads(match.group(0))

    if not isinstance(data, dict):
        raise ValueError("Decomposer output JSON is not an object.")

    return data