# src/scarecrow/runtime/observation.py

"""LLM-based Observation Extraction.

每轮 Agent 执行后，从本轮 ToolMessage / AIMessage 中抽取结构化观察，
写入当前 SessionState.task_state。

这一层只负责抽取“可复用的短期任务状态”，不是长期记忆。
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
      "alias": "dataset_alias",
      "path": "relative/path/to/data_file.csv",
      "description": "简短描述该数据集的用途和主要字段"
    }
  ],
  "findings": [
    {
      "source": "relative/path/to/data_file.csv",
      "summary": "明确由本轮工具结果支持的可复用发现",
      "confidence": 0.9
    }
  ]
}

抽取规则：

1. datasets
当本轮消息明确出现数据文件路径、文件名、列名、表用途时，抽取 dataset。
- alias 通常使用文件 stem，例如 relative/path/to/data_file.csv -> data_file
- path 必须是消息中明确出现的相对路径或绝对路径
- description 只能根据本轮消息中出现的列名、文件名、工具结果简短描述，不要编造
- description 应该短，不要超过 80 个中文字符

2. findings
当本轮消息明确出现以下结论时，抽取 finding：
- 缺失值
- 异常值
- 外键/引用完整性问题
- 离群记录
- 字段类型问题
- 数据清洗建议
- 统计分析结果
- 重要业务结论

3. Finding 质量要求
- finding 必须是后续任务可复用的概括性结论，不是逐行日志。
- 不要记录过细的逐行事实，例如“第 N 行某个具体实体的值为空”，除非用户明确要求追踪该实体。
- 如果只是预览样本中发现缺失，写成“某列在预览样本中出现缺失值”。
- 如果是全量分析结果，写成“某列存在 N 个缺失值，占比 X%”。
- 如果是异常值，写成“某列存在异常高值/低值，异常值为 X”。
- summary 应该短、准、可复用，通常不超过 80 个中文字符。
- confidence 必须反映证据强度：工具明确输出的事实用 0.85-1.0；模型推测不要写入。

4. 不要抽取：
- 普通寒暄
- 工具调用失败但没有实质发现
- 用户没有确认的猜测
- 没有证据的推断
- 只是“可以继续分析吗”这种建议
- 示例中的占位符本身
- 没有后续复用价值的临时表述

5. 如果本轮没有可复用状态，输出：
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

    cleaned_update = sanitize_observation_update(update)
    apply_observation_update(task_state, cleaned_update)
    return cleaned_update


def sanitize_observation_update(update: ObservationUpdate) -> ObservationUpdate:
    """清理 LLM 抽取结果，避免把低质量 observation 写入 TaskState。"""

    datasets: list[ObservationDataset] = []
    findings: list[ObservationFinding] = []

    for dataset in update.datasets:
        alias = _normalize_alias(dataset.alias)
        path = _normalize_source(dataset.path)
        description = _compact_text(dataset.description, max_chars=120)

        if not alias or not path:
            continue

        if _looks_like_placeholder(alias) or _looks_like_placeholder(path):
            continue

        datasets.append(
            ObservationDataset(
                alias=alias,
                path=path,
                description=description,
            )
        )

    for finding in update.findings:
        source = _normalize_source(finding.source)
        summary = _compact_text(finding.summary, max_chars=160)
        confidence = finding.confidence

        if confidence < 0.6:
            continue

        if len(summary) < 8:
            continue

        if _looks_like_placeholder(source) or _looks_like_placeholder(summary):
            continue

        if _looks_like_overly_specific_row_log(summary):
            summary = _generalize_row_level_summary(summary)

        findings.append(
            ObservationFinding(
                source=source or "observation",
                summary=summary,
                confidence=confidence,
            )
        )

    return ObservationUpdate(datasets=datasets, findings=findings)


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
            blocks.append(f"[ToolMessage:{name}]\n{_truncate(content, 6000)}")

        elif isinstance(msg, AIMessage):
            content = _message_content_to_text(msg.content)
            if content:
                blocks.append(f"[AIMessage]\n{_truncate(content, 6000)}")

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


def _normalize_alias(alias: str) -> str:
    """规范化 dataset alias。"""

    cleaned = alias.strip().strip("`'\"")
    cleaned = cleaned.replace("\\", "/")

    if "/" in cleaned:
        cleaned = cleaned.rsplit("/", 1)[-1]

    if "." in cleaned:
        cleaned = cleaned.rsplit(".", 1)[0]

    cleaned = re.sub(r"[^A-Za-z0-9_\-]+", "_", cleaned)
    cleaned = cleaned.strip("_-")

    return cleaned


def _normalize_source(source: str) -> str:
    """规范化 source/path，尽量避免绝对路径污染 TaskState。"""

    cleaned = source.strip().strip("`'\"")
    cleaned = cleaned.replace("\\", "/")

    marker = "/data/"
    if marker in cleaned:
        return "data/" + cleaned.split(marker, 1)[1]

    marker = "/src/"
    if marker in cleaned:
        return "src/" + cleaned.split(marker, 1)[1]

    return cleaned


def _compact_text(text: str, max_chars: int) -> str:
    """压缩多余空白并截断文本。"""

    cleaned = " ".join(text.strip().split())

    if len(cleaned) <= max_chars:
        return cleaned

    return cleaned[: max_chars - 3] + "..."


def _looks_like_placeholder(text: str) -> bool:
    """过滤 prompt 示例里的占位符。"""

    lowered = text.lower()

    placeholders = [
        "dataset_alias",
        "relative/path/to",
        "data_file.csv",
        "明确由本轮工具结果支持",
        "简短描述该数据集",
    ]

    return any(item in lowered for item in placeholders)


def _looks_like_overly_specific_row_log(summary: str) -> bool:
    """判断 finding 是否过于像逐行日志。"""

    patterns = [
        r"第\s*\d+\s*行",
        r"row\s*\d+",
        r"index\s*\d+",
    ]

    return any(re.search(pattern, summary, flags=re.IGNORECASE) for pattern in patterns)


def _generalize_row_level_summary(summary: str) -> str:
    """把过细的逐行 finding 轻量泛化。

    注意：这里不做业务猜测，只去掉行级表述倾向。
    真正概括仍主要交给 LLM prompt 控制。
    """

    summary = re.sub(r"第\s*\d+\s*行", "某行", summary)
    summary = re.sub(r"row\s*\d+", "some row", summary, flags=re.IGNORECASE)
    summary = re.sub(r"index\s*\d+", "some index", summary, flags=re.IGNORECASE)

    return summary


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text

    return text[:max_chars] + "\n\n[... truncated for observation extraction ...]"