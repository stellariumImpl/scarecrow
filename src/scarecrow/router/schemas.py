from typing import Literal

from pydantic import BaseModel, Field

IntentName = Literal[
    "chat",
    "data_analysis",
    "file_inspection",
    "code_debugging",
    "config",
    "unknown",
]

RiskLevel = Literal["low", "medium", "high"]


class RouteDecision(BaseModel):
    """Intent Router 的结构化输出。

    Router 只负责判断用户请求类型，不负责执行任务。
    """

    intent: IntentName = Field(
        ...,
        description="用户请求的主要意图。",
    )

    confidence: float = Field(
        ...,
        ge=0,
        le=1,
        description="路由置信度，范围 0 到 1。",
    )

    required_skills: list[str] = Field(
        default_factory=list,
        description="建议加载的 skill 名称。",
    )

    required_tools: list[str] = Field(
        default_factory=list,
        description="建议允许使用的 tool 名称。",
    )

    needs_clarification: bool = Field(
        default=False,
        description="是否需要先向用户澄清。",
    )

    clarification_question: str | None = Field(
        default=None,
        description="需要澄清时给用户的问题。",
    )

    risk_level: RiskLevel = Field(
        default="low",
        description="执行该请求的风险等级。",
    )

    reason: str = Field(
        default="",
        description="简短说明为什么这样路由。不要输出长推理。",
    )
