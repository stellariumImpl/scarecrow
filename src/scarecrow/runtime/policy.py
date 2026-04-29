# src/scarecrow/runtime/policy.py

from dataclasses import dataclass
from typing import Literal

from scarecrow.router import RouteDecision


PolicyAction = Literal[
    "run_agent",
    "show_message",
]


@dataclass(frozen=True)
class RuntimePolicyDecision:
    """Runtime 层的执行策略决策。

    Intent Router 只负责判断用户意图。
    Runtime Policy 负责决定这个意图是否应该进入 Agent。
    """

    action: PolicyAction
    message: str | None = None


def decide_runtime_policy(decision: RouteDecision) -> RuntimePolicyDecision:
    """根据 RouteDecision 决定是否进入 Agent。"""

    if decision.needs_clarification and decision.clarification_question:
        return RuntimePolicyDecision(
            action="show_message",
            message=decision.clarification_question,
        )

    if decision.intent == "config":
        return RuntimePolicyDecision(
            action="show_message",
            message=(
                "这是模型或系统配置类请求。请使用 `/config` 修改 provider / model / API key；"
                "如需配置 LangSmith，请使用 `/langsmith`。"
            ),
        )

    if decision.intent == "unknown":
        return RuntimePolicyDecision(
            action="show_message",
            message="我还不确定你想让我做什么。请补充任务目标、文件名或期望输出。",
        )

    if decision.risk_level == "high":
        return RuntimePolicyDecision(
            action="show_message",
            message=(
                "这个请求风险较高，我不会直接执行。"
                "请明确说明你的目标、影响范围，以及是否允许执行相关操作。"
            ),
        )

    return RuntimePolicyDecision(action="run_agent")