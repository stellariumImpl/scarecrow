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


def select_skills_from_decision(decision: RouteDecision) -> list[str]:
    """根据 RouteDecision 选择本轮要注入的 Skill。

    Skill 由 Router 建议，Runtime 只做去重。
    """

    return list(dict.fromkeys(decision.required_skills))


def select_tools_from_decision(decision: RouteDecision) -> list[str]:
    """根据 RouteDecision.required_capabilities 选择本轮可暴露的 Tool。

    Runtime 不写死 intent/task 到具体工具的映射。
    """

    tool_registry = build_default_tool_registry()

    # 新逻辑：capability-driven。
    selected_tools = tool_registry.select_tool_names_by_capabilities(
        required_capabilities=decision.required_capabilities,
        max_risk=decision.risk_level,
    )

    # 过渡兼容：如果 Router 还输出 required_tools，则合并进去。
    # 后续稳定后可以删除这一段。
    for tool_name in decision.required_tools:
        if tool_name not in selected_tools:
            selected_tools.append(tool_name)

    return selected_tools


def _looks_like_preview_request(decision: RouteDecision) -> bool:
    """判断 data_analysis 请求是否更像安全预览而不是复杂分析。

    这里先基于 Router 给出的工具/skill 做保守判断。
    后面如果 RouteDecision 增加 task_type，可以替换掉这个启发式。
    """

    if "preview_data_file" in decision.required_tools:
        return True

    if "run_python" in decision.required_tools:
        return False

    # 没有复杂 skill 时，倾向于允许 preview。
    return "data-explorer" not in decision.required_skills

def build_agent(
    cfg: LLMConfig,
    workspace: Path,
    selected_skills: list[str] | None = None,
    selected_tools: list[str] | None = None,
    include_all_skills: bool = False,
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

def _is_duplicate_tool_call(msg, seen_tool_calls: set[str]) -> str | None:
    """检测同一轮中是否重复调用同一个工具和同一组参数。

    返回重复调用签名；没有重复则返回 None。
    """

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


def _build_duplicate_tool_warning(signature: str):
    """构造一个轻量 AIMessage，提示工具调用已被中断。"""

    from langchain_core.messages import AIMessage

    return AIMessage(
        content=(
            "检测到重复工具调用，已停止本轮继续执行，避免无意义循环。\n\n"
            f"重复调用: `{signature}`\n\n"
            "请确认要查找的文件名或换一个更明确的描述。"
        )
    )

def stream_agent_response(
    state: SessionState,
    user_input: str,
) -> Iterator[Any]:
    """向当前 agent 发送用户输入，并流式返回新增消息。

    这个函数只负责执行，不负责渲染。
    同一轮中会检测重复 tool call，避免 Agent 陷入无意义循环。
    """

    if state.agent is None:
        raise RuntimeError("Agent is not initialized.")

    state.messages.append({"role": "user", "content": user_input})

    result_messages = list(state.messages)
    seen_tool_calls: set[str] = set()

    try:
        for chunk in state.agent.stream(
            {"messages": state.messages},
            stream_mode="values",
            # config={"recursion_limit": 8},
        ):
            new_messages = chunk.get("messages", [])

            for msg in new_messages[len(result_messages):]:
                duplicate = _is_duplicate_tool_call(msg, seen_tool_calls)
                if duplicate:
                    yield _build_duplicate_tool_warning(duplicate)
                    state.messages = result_messages
                    return

                yield msg

            result_messages = new_messages

        state.messages = result_messages

    except Exception:
        if state.messages:
            state.messages.pop()
        raise