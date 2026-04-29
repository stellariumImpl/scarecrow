# src/scarecrow/runtime/agent.py

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from langchain.agents import create_agent

from scarecrow.config import LLMConfig, SKILLS_DIR
from scarecrow.context import build_system_prompt
from scarecrow.llm import load_chat_model_from_config
from scarecrow.router import IntentRouter, RouteDecision
from scarecrow.runtime.state import SessionState
from scarecrow.tools import build_default_tool_registry


def route_user_input(user_input: str, cfg: LLMConfig) -> RouteDecision:
    """对用户输入做结构化路由。"""

    model = load_chat_model_from_config(cfg)
    router = IntentRouter(model)
    return router.route(user_input)


def select_skills_from_decision(decision: RouteDecision) -> list[str]:
    """根据 RouteDecision 选择本轮要注入的 Skill。"""

    selected_skills = list(decision.required_skills)

    if decision.intent == "data_analysis" and "run-python" not in selected_skills:
        selected_skills.append("run-python")

    return selected_skills


def select_tools_from_decision(decision: RouteDecision) -> list[str]:
    """根据 RouteDecision 选择本轮可暴露的 Tool。"""

    selected_tools = list(decision.required_tools)

    if decision.intent == "data_analysis" and "run_python" not in selected_tools:
        selected_tools.append("run_python")

    return selected_tools


def build_agent(
    cfg: LLMConfig,
    workspace: Path,
    selected_skills: list[str] | None = None,
    selected_tools: list[str] | None = None,
    include_all_skills: bool = False,
):
    """按配置、工作区、skills 和 tools 构建 Agent。"""

    model = load_chat_model_from_config(cfg)

    tool_registry = build_default_tool_registry()
    tools = tool_registry.select_tools(selected_tools or [])

    return create_agent(
        model=model,
        tools=tools,
        system_prompt=build_system_prompt(
            workspace=workspace,
            selected_skills=selected_skills,
            include_all_skills=include_all_skills,
            include_skill_index=False,
            skills_dir=SKILLS_DIR,
        ),
    )


def prepare_agent_for_message(
    cfg: LLMConfig,
    state: SessionState,
    decision: RouteDecision,
):
    """根据路由结果构建本轮 Agent。"""

    selected_skills = select_skills_from_decision(decision)
    selected_tools = select_tools_from_decision(decision)

    agent = build_agent(
        cfg=cfg,
        workspace=state.workspace,
        selected_skills=selected_skills,
        selected_tools=selected_tools,
        include_all_skills=False,
    )

    state.agent = agent

    return agent, selected_skills, selected_tools


def stream_agent_response(
    state: SessionState,
    user_input: str,
) -> Iterator[Any]:
    """向当前 agent 发送用户输入，并流式返回新增消息。

    这个函数只负责执行，不负责渲染。
    """

    if state.agent is None:
        raise RuntimeError("Agent is not initialized.")

    state.messages.append({"role": "user", "content": user_input})

    result_messages = list(state.messages)

    try:
        for chunk in state.agent.stream(
            {"messages": state.messages},
            stream_mode="values",
        ):
            new_messages = chunk.get("messages", [])

            for msg in new_messages[len(result_messages):]:
                yield msg

            result_messages = new_messages

        state.messages = result_messages

    except Exception:
        if state.messages:
            state.messages.pop()
        raise