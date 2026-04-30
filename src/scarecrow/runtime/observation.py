# src/scarecrow/runtime/observation.py

"""LLM-based Observation Extraction.

每轮 Agent 执行后，从本轮 ToolMessage / AIMessage 中抽取结构化观察，
写入当前 SessionState.task_state。
"""

import json
import re
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field, ValidationError

from scarecrow.runtime.task_state import TaskState


class ObservationDataset(BaseModel):
    """LLM 抽取出的数据集记录。"""

    alias: str = Field(default="")
    path: str = Field(default="")
    description: str = Field(default="")


class ObservationFinding(BaseModel):
    """LLM 抽取出的分析结论。"""

    source: str = Field(default="")
    summary: str = Field(default="")
    confidence: float = Field(default=0.8, ge=0, le=1)


class ObservationUpdate(BaseModel):
    """一轮对话产生的任务状态更新。"""

    datasets: list[ObservationDataset] = Field(default_factory=list)
    findings: list[ObservationFinding] = Field(default_factory=list)


_OBSERVATION_SYSTEM_PROMPT = """你是 Scarecrow 的 Observation Extractor。

你的职责：
- 从本轮 Agent 的工具结果和最终回答中抽取“可复用的短期任务状态”
- 输出严格 JSON
- 不回答用户问题
- 不编造事实
- 不输出 markdown
- 不输出 ```json 代码块
- 只抽取本轮消息中明确支持的事实

你必须输出如下 JSON 结构：

{
  "datasets": [
    {
      "alias": "users",
      "path": "data/users.csv",
      "description": "用户基础信息表，包含 user_id/name/age/city/signup_date"
    }
  ],
  "findings": [
    {
      "source": "data/users.csv",
      "summary": "age 存在 1 个缺失值，age=120 是异常值",
      "confidence": 0.9
    }
  ]
}

抽取规则：

1. datasets
当本轮消息明确出现数据文件路径、文件名、列名、表用途时，抽取 dataset。
- alias 通常使用文件 stem，例如 data/users.csv -> users
- path 必须是消息中明确出现的路径，例如 data/users.csv
- description 简短描述该数据集，不要编造

2. findings
当本轮消息明确出现以下结论时，抽取 finding：
- 缺失值
- 异常值
- 外键/引用完整性问题
- 离群订单
- 字段类型问题
- 数据清洗建议
- 统计分析结果
- 重要业务结论

3. 不要抽取：
- 普通寒暄
- 工具调用失败但没有实质发现
- 用户没有确认的猜测
- 没有证据的推断
- 只是“可以继续分析吗”这种建议

4. 如果本轮没有可复用状态，输出：
{
  "datasets": [],
  "findings": []
}
"""


def extract_observations_with_llm(
    model: BaseChatModel,
    messages: list[Any],
    task_state: TaskState,
) -> ObservationUpdate:
    """从本轮新增消息中抽取 observation，并写入 task_state。

    失败时返回空更新，不影响主对话。
    """

    transcript = _messages_to_observation_text(messages)

    if not transcript.strip():
        return ObservationUpdate()

    user_prompt = f"""当前已有任务状态：

{task_state.brief()}

本轮新增消息：

{transcript}

请从“本轮新增消息”中抽取可复用的短期任务状态，输出严格 JSON。
"""

    try:
        msg = model.invoke(
            [
                SystemMessage(content=_OBSERVATION_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]
        )
        content = _message_content_to_text(msg.content)
        data = _extract_json_object(content)
        update = ObservationUpdate.model_validate(data)
    except (json.JSONDecodeError, ValueError, ValidationError, Exception):
        return ObservationUpdate()

    apply_observation_update(task_state, update)
    return update


def apply_observation_update(
    task_state: TaskState,
    update: ObservationUpdate,
) -> None:
    """把 ObservationUpdate 写入 TaskState。"""

    for dataset in update.datasets:
        if not dataset.alias or not dataset.path:
            continue

        task_state.add_dataset(
            alias=dataset.alias,
            path=dataset.path,
            description=dataset.description,
        )

    for finding in update.findings:
        if not finding.summary:
            continue

        task_state.add_finding(
            source=finding.source or "observation",
            summary=finding.summary,
            confidence=finding.confidence,
        )


def _messages_to_observation_text(messages: list[Any]) -> str:
    """把本轮新增消息压缩成 Observation Extractor 可读文本。"""

    blocks: list[str] = []

    for msg in messages:
        if isinstance(msg, ToolMessage):
            name = getattr(msg, "name", None) or "tool"
            content = str(msg.content) if msg.content else ""
            blocks.append(
                f"[ToolMessage:{name}]\n{_truncate(content, 6000)}"
            )

        elif isinstance(msg, AIMessage):
            content = _message_content_to_text(msg.content)
            if content:
                blocks.append(
                    f"[AIMessage]\n{_truncate(content, 6000)}"
                )

    return "\n\n".join(blocks)


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
        raise ValueError(f"No JSON object found in observation output: {text}")

    data = json.loads(match.group(0))

    if not isinstance(data, dict):
        raise ValueError("Observation output JSON is not an object.")

    return data


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text

    return text[:max_chars] + "\n\n[... truncated for observation extraction ...]"