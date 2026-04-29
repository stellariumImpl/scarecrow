# src/scarecrow/router/schemas.py

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

    Router 只负责判断用户请求需要哪些能力，不负责选择具体工具。
    具体工具由 ToolRegistry 根据 required_capabilities 自动匹配。
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

    required_capabilities: list[str] = Field(
        default_factory=list,
        description=(
            "完成该请求所需的抽象能力，而不是具体工具名。"
            "例如 workspace.resolve_path、data.preview、python.execute。"
        ),
    )

    required_skills: list[str] = Field(
        default_factory=list,
        description="建议加载的 skill 名称。Skill 仍可由 Router 建议，但不直接等同于工具。",
    )

    # 过渡字段：短期保留，避免旧代码或旧测试立刻断裂。
    # 新逻辑不应依赖它；后续稳定后可以删除。
    required_tools: list[str] = Field(
        default_factory=list,
        description="过渡字段：旧版工具名建议。新逻辑优先使用 required_capabilities。",
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