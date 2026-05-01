# src/scarecrow/runtime/agent.py

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage

from scarecrow.config import LLMConfig, SKILLS_DIR
from scarecrow.context import build_system_prompt
from scarecrow.llm import load_chat_model_from_config
from scarecrow.router import IntentRouter, RouteDecision
from scarecrow.runtime.observation import extract_observations_with_llm
from scarecrow.runtime.state import SessionState
from scarecrow.skills import load_skill_registry
from scarecrow.tools import (
    build_default_tool_registry,
    set_preview_workspace,
    set_workspace,
)


def route_user_input(user_input: str, cfg: LLMConfig) -> RouteDecision:
    """对用户输入做结构化路由。"""

    model = load_chat_model_from_config(cfg)
    tool_registry = build_default_tool_registry()
    capability_index = tool_registry.build_capability_index()

    router = IntentRouter(model, capability_index=capability_index)
    return router.route(user_input)


def select_skills_from_decision(
    decision: RouteDecision,
    skills_dir: Path = SKILLS_DIR,
) -> list[str]:
    """根据 capabilities 和 Router 建议选择本轮要注入的 Skill。

    渐进式策略：
    1. 先根据 required_capabilities 自动匹配声明了 capabilities 的 skill
    2. 再合并 Router.required_skills
    3. 去重并保持顺序
    """

    skill_registry = load_skill_registry(skills_dir)

    selected: list[str] = []

    selected.extend(
        skill_registry.select_skill_names_by_capabilities(
            decision.required_capabilities
        )
    )

    selected.extend(decision.required_skills)

    return list(dict.fromkeys(selected))


def select_tools_from_decision(decision: RouteDecision) -> list[str]:
    """根据 RouteDecision.required_capabilities 选择本轮可暴露的 Tool。

    Runtime 不写死 intent/task 到具体工具的映射。
    """

    tool_registry = build_default_tool_registry()

    known_capabilities, _ = tool_registry.validate_capabilities(
        decision.required_capabilities
    )

    selected_tools = tool_registry.select_tool_names_by_capabilities(
        required_capabilities=known_capabilities,
        max_risk=decision.risk_level,
    )

    # 过渡兼容：如果旧字段 required_tools 仍有内容，则合并进去。
    for tool_name in decision.required_tools:
        if tool_name not in selected_tools:
            selected_tools.append(tool_name)

    return selected_tools


def inspect_capability_selection(
    decision: RouteDecision,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """返回 capability 诊断信息。

    返回：
    - known_capabilities: Tool 或 Skill 支持的能力
    - unknown_capabilities: Tool 和 Skill 都不支持的能力
    - selected_tools: 根据 Tool capability 选出的工具
    - selected_skills: 根据 Skill capability 选出的技能
    """

    tool_registry = build_default_tool_registry()
    skill_registry = load_skill_registry(SKILLS_DIR)

    tool_known, _ = tool_registry.validate_capabilities(
        decision.required_capabilities
    )
    skill_known, _ = skill_registry.validate_capabilities(
        decision.required_capabilities
    )

    known_capabilities = list(dict.fromkeys(tool_known + skill_known))

    unknown_capabilities = [
        capability
        for capability in decision.required_capabilities
        if capability not in known_capabilities
    ]

    selected_tools = tool_registry.select_tool_names_by_capabilities(
        required_capabilities=tool_known,
        max_risk=decision.risk_level,
    )

    for tool_name in decision.required_tools:
        if tool_name not in selected_tools:
            selected_tools.append(tool_name)

    selected_skills = skill_registry.select_skill_names_by_capabilities(
        decision.required_capabilities
    )

    for skill_name in decision.required_skills:
        if skill_name not in selected_skills:
            selected_skills.append(skill_name)

    return (
        known_capabilities,
        list(dict.fromkeys(unknown_capabilities)),
        selected_tools,
        selected_skills,
    )

def build_agent(
    cfg: LLMConfig,
    workspace: Path,
    selected_skills: list[str] | None = None,
    selected_tools: list[str] | None = None,
    include_all_skills: bool = False,
    task_state_brief: str | None = None,
):
    """按配置、工作区、skills 和 tools 构建 Agent。"""

    model = load_chat_model_from_config(cfg)

    set_workspace(workspace)
    set_preview_workspace(workspace)

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
            task_state_brief=task_state_brief,
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
        task_state_brief=state.task_state.brief(),
    )

    state.agent = agent

    return agent, selected_skills, selected_tools


def stream_agent_response(
    cfg: LLMConfig,
    state: SessionState,
    user_input: str,
) -> Iterator[Any]:
    """向当前 agent 发送用户输入，并流式返回新增消息。

    这个函数只负责执行，不负责渲染。
    同一轮中会检测重复 tool call，避免 Agent 陷入无意义循环。
    执行成功后，会用 LLM 自动从本轮消息中抽取 observations 写入 task_state。
    """

    if state.agent is None:
        raise RuntimeError("Agent is not initialized.")

    state.messages.append({"role": "user", "content": user_input})

    result_messages = list(state.messages)
    emitted_messages: list[Any] = []
    seen_tool_calls: set[str] = set()

    try:
        for chunk in state.agent.stream(
            {"messages": state.messages},
            stream_mode="values",
            config={"recursion_limit": 8},
        ):
            new_messages = chunk.get("messages", [])

            for msg in new_messages[len(result_messages):]:
                duplicate = _is_duplicate_tool_call(msg, seen_tool_calls)
                if duplicate:
                    warning = _build_duplicate_tool_warning(duplicate)
                    emitted_messages.append(warning)
                    yield warning
                    state.messages = result_messages
                    return

                emitted_messages.append(msg)
                yield msg

            result_messages = new_messages

        state.messages = result_messages

        if emitted_messages:
            observation_model = load_chat_model_from_config(cfg)
            extract_observations_with_llm(
                model=observation_model,
                messages=emitted_messages,
                task_state=state.task_state,
            )

    except Exception:
        if state.messages:
            state.messages.pop()
        raise


def _is_duplicate_tool_call(msg, seen_tool_calls: set[str]) -> str | None:
    """检测同一轮中是否重复调用同一个工具和同一组参数。"""

    tool_calls = getattr(msg, "tool_calls", None)
    if not tool_calls:
        return None

    for tool_call in tool_calls:
        name = tool_call.get("name", "")
        args = tool_call.get("args", {}) or {}

        signature = _tool_call_signature(name, args)

        if signature in seen_tool_calls:
            return signature

        seen_tool_calls.add(signature)

    return None


def _tool_call_signature(name: str, args: dict) -> str:
    """生成稳定的 tool call 签名。"""

    import json

    try:
        args_text = json.dumps(args, sort_keys=True, ensure_ascii=False)
    except TypeError:
        args_text = str(args)

    return f"{name}:{args_text}"


def _build_duplicate_tool_warning(signature: str) -> AIMessage:
    """构造一个轻量 AIMessage，提示工具调用已被中断。"""

    return AIMessage(
        content=(
            "检测到重复工具调用，已停止本轮继续执行，避免无意义循环。\n\n"
            f"重复调用: `{signature}`\n\n"
            "请确认要查找的文件名或换一个更明确的描述。"
        )
    )