# src/scarecrow/repl.py

"""Scarecrow REPL - 交互式对话循环"""

import json
from pathlib import Path

from langchain_core.messages import AIMessage, ToolMessage
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from rich.console import Console

from scarecrow.config import (
    CONFIG_DIR,
    CONFIG_FILE,
    SKILLS_DIR,
    LangSmithConfig,
    LLMConfig,
    PROVIDER_LABELS,
    PROVIDER_MODELS,
    clear_langsmith_config,
    ensure_config_dir,
    load_config,
    load_langsmith_config,
    save_config,
    save_langsmith_config,
)
from scarecrow.langsmith_setup import apply_langsmith_env
from scarecrow.runtime import (
    PlanStep,
    SessionState,
    TaskPlan,
    decide_runtime_policy,
    decompose_user_input,
    inspect_capability_selection,
    plan_user_input,
    prepare_agent_for_message,
    route_user_input,
    stream_agent_response,
)
from scarecrow.skills import ensure_builtin_skills
from scarecrow.tools import reset_namespace


console = Console()


class _Cancelled(Exception):
    """用户取消配置流程。"""


def start_repl(workspace: Path) -> None:
    """启动 REPL 循环。"""

    ensure_config_dir()
    SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    ensure_builtin_skills(SKILLS_DIR)

    ls_enabled, ls_project = apply_langsmith_env()

    history_file = CONFIG_DIR / "history.txt"
    session: PromptSession = PromptSession(history=FileHistory(str(history_file)))

    console.print(f"Scarecrow — 当前工作区: {workspace}")
    console.print(f"[dim]Skills 目录: {SKILLS_DIR}[/dim]")

    if ls_enabled:
        console.print(f"[dim]LangSmith: 已启用 (project: {ls_project})[/dim]")
    else:
        console.print("[dim]LangSmith: 未启用 (输入 /langsmith 配置)[/dim]")

    console.print("\n输入 /help 查看命令，/quit 退出")

    state = SessionState(workspace=workspace)

    while True:
        try:
            user_input = session.prompt(
                HTML("<ansicyan>Scarecrow > </ansicyan>")
            ).strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n再见！")
            break

        if not user_input:
            continue

        if user_input == "/quit":
            console.print("再见！")
            break

        if user_input == "/help":
            _show_help()
            continue

        if user_input == "/config":
            _do_config(session)
            state.agent = None
            continue

        if user_input == "/langsmith":
            _do_langsmith(session)
            apply_langsmith_env()
            state.agent = None
            continue

        if user_input == "/reset":
            state.reset()
            reset_namespace()
            console.print("[dim]已清空对话历史、任务状态与 Python 命名空间[/dim]")
            continue

        if user_input == "/state":
            console.print(state.task_state.brief(), markup=False)
            continue

        if user_input == "/plan-mode":
            _show_plan_mode(state)
            continue

        if user_input == "/plan-mode on":
            state.plan_mode = True
            console.print("[green]✓ Plan Mode 已开启[/green]")
            continue

        if user_input == "/plan-mode off":
            state.plan_mode = False
            console.print("[green]✓ Plan Mode 已关闭[/green]")
            continue

        if user_input.startswith("/route "):
            _debug_route(user_input.removeprefix("/route ").strip())
            continue

        if user_input.startswith("/decompose "):
            _debug_decompose(
                user_input.removeprefix("/decompose ").strip(),
                state,
            )
            continue

        if user_input.startswith("/plan "):
            _debug_plan(
                user_input.removeprefix("/plan ").strip(),
                state,
            )
            continue

        _handle_chat(user_input, state)


# ---------------------------------------------------------------------------
# Debug commands
# ---------------------------------------------------------------------------


def _debug_route(text: str) -> None:
    """调试 Intent Router。"""

    if not text:
        console.print("[yellow]用法: /route <用户请求>[/yellow]")
        return

    cfg = load_config()
    if cfg is None:
        console.print("[yellow]请先运行 /config 配置 LLM[/yellow]")
        return

    try:
        with console.status("Routing..."):
            decision = route_user_input(text, cfg)
    except Exception as e:
        console.print("[red]Router 出错:[/red]", end=" ")
        console.print(str(e), markup=False)
        return

    _print_json(decision.model_dump(mode="json"))


def _debug_decompose(text: str, state: SessionState) -> None:
    """调试 Task Decomposer。"""

    if not text:
        console.print("[yellow]用法: /decompose <用户请求>[/yellow]")
        return

    cfg = load_config()
    if cfg is None:
        console.print("[yellow]请先运行 /config 配置 LLM[/yellow]")
        return

    try:
        with console.status("Decomposing..."):
            result = decompose_user_input(
                user_input=text,
                cfg=cfg,
                task_state_brief=state.task_state.brief(),
            )
    except Exception as e:
        console.print("[red]Decomposer 出错:[/red]", end=" ")
        console.print(str(e), markup=False)
        return

    _print_json(result.model_dump(mode="json"))


def _debug_plan(text: str, state: SessionState) -> None:
    """调试 Task Planner。"""

    if not text:
        console.print("[yellow]用法: /plan <用户请求>[/yellow]")
        return

    cfg = load_config()
    if cfg is None:
        console.print("[yellow]请先运行 /config 配置 LLM[/yellow]")
        return

    try:
        with console.status("Planning..."):
            plan = plan_user_input(
                user_input=text,
                cfg=cfg,
                task_state_brief=state.task_state.brief(),
            )
    except Exception as e:
        console.print("[red]Planner 出错:[/red]", end=" ")
        console.print(str(e), markup=False)
        return

    _print_json(plan.model_dump(mode="json"))


def _show_plan_mode(state: SessionState) -> None:
    """显示当前 Plan Mode 状态。"""

    status = "on" if state.plan_mode else "off"
    console.print(f"[cyan]Plan Mode: {status}[/cyan]")


def _print_json(data: dict) -> None:
    """安全打印 JSON。"""

    console.print_json(json.dumps(data, ensure_ascii=False, indent=2))


# ---------------------------------------------------------------------------
# Chat execution
# ---------------------------------------------------------------------------


def _handle_chat(user_input: str, state: SessionState) -> None:
    """处理一轮用户输入。

    Plan Mode OFF:
        走现有 decomposer + single task 执行链路。

    Plan Mode ON:
        走 Task Planner + PlanStep 顺序执行链路。
    """

    cfg = load_config()
    if cfg is None:
        console.print("[yellow]请先运行 /config 配置 LLM[/yellow]")
        return

    if state.plan_mode:
        _handle_planned_chat(user_input, state, cfg)
        return

    _handle_decomposed_chat(user_input, state, cfg)


def _handle_decomposed_chat(
    user_input: str,
    state: SessionState,
    cfg: LLMConfig,
) -> None:
    """Plan Mode OFF 时的默认执行路径。

    保持现有行为：
    - 单任务直接执行
    - 显式多任务按 Decomposer 拆分后顺序执行
    """

    try:
        with console.status("Decomposing..."):
            decomposition = decompose_user_input(
                user_input=user_input,
                cfg=cfg,
                task_state_brief=state.task_state.brief(),
            )
    except Exception as e:
        console.print("[red]Decomposer 出错:[/red]", end=" ")
        console.print(str(e), markup=False)
        return

    if not decomposition.is_multi_task:
        _handle_single_chat(user_input, state, cfg=cfg)
        return

    console.print()
    console.print(
        f"检测到多任务，拆分为 {len(decomposition.tasks)} 个子任务：",
        style="bold cyan",
    )

    for i, task in enumerate(decomposition.tasks, start=1):
        console.print(f"{i}. {task.text}", style="cyan", markup=False)

    for i, task in enumerate(decomposition.tasks, start=1):
        console.print()
        console.print(
            f"子任务 {i}/{len(decomposition.tasks)}",
            style="bold cyan",
        )
        console.print(task.text, style="cyan", markup=False)

        ok = _handle_single_chat(task.text, state, cfg=cfg)

        if not ok:
            console.print(
                f"子任务 {i} 执行失败，已停止后续子任务。",
                style="yellow",
            )
            return

    console.print()
    console.print("✓ 多任务执行完成", style="green")


def _handle_planned_chat(
    user_input: str,
    state: SessionState,
    cfg: LLMConfig,
) -> None:
    """Plan Mode ON 时的执行路径。"""

    try:
        with console.status("Planning..."):
            plan = plan_user_input(
                user_input=user_input,
                cfg=cfg,
                task_state_brief=state.task_state.brief(),
            )
    except Exception as e:
        console.print("[red]Planner 出错:[/red]", end=" ")
        console.print(str(e), markup=False)
        return

    _render_plan_summary(plan)

    if plan.requires_confirmation:
        console.print("[yellow]该计划需要确认后才能执行。[/yellow]")
        return

    if not plan.steps:
        console.print("[yellow]Planner 没有生成可执行步骤。[/yellow]")
        return

    for index, step in enumerate(plan.steps, start=1):
        console.print()
        console.print(
            f"计划步骤 {index}/{len(plan.steps)}: {step.id}",
            style="bold cyan",
        )
        console.print(step.instruction, style="cyan", markup=False)

        if step.requires_user_input:
            question = step.user_question or "该步骤需要补充信息后才能执行。"
            console.print("[yellow]需要用户补充信息：[/yellow]", end=" ")
            console.print(question, markup=False)
            return

        step_input = build_step_execution_input(step)
        ok = _handle_single_chat(
            step_input,
            state,
            cfg=cfg,
            isolated_messages=True,
        )

        if not ok:
            console.print(
                f"计划步骤 {index} 执行失败，已停止后续步骤。",
                style="yellow",
            )
            return

    console.print()
    console.print("✓ Plan 执行完成", style="green")


def _handle_single_chat(
    user_input: str,
    state: SessionState,
    cfg: LLMConfig | None = None,
    isolated_messages: bool = False,
) -> bool:
    """处理单个任务。

    返回：
    - True：执行成功
    - False：执行失败或被 policy 拦截
    """

    if cfg is None:
        cfg = load_config()

    if cfg is None:
        console.print("[yellow]请先运行 /config 配置 LLM[/yellow]")
        return False

    try:
        with console.status("Routing..."):
            decision = route_user_input(
                user_input=user_input,
                cfg=cfg,
                task_state_brief=state.task_state.brief(),
            )
    except TypeError:
        # 兼容旧版本 route_user_input(user_input, cfg)。
        try:
            with console.status("Routing..."):
                decision = route_user_input(user_input, cfg)
        except Exception as e:
            console.print("[red]Router 出错:[/red]", end=" ")
            console.print(str(e), markup=False)
            return False
    except Exception as e:
        console.print("[red]Router 出错:[/red]", end=" ")
        console.print(str(e), markup=False)
        return False

    policy_decision = decide_runtime_policy(decision)

    if policy_decision.action == "show_message":
        if policy_decision.message:
            console.print(f"[yellow]{policy_decision.message}[/yellow]")
        return False

    try:
        (
            known_capabilities,
            unknown_capabilities,
            inspected_tools,
            inspected_skills,
        ) = inspect_capability_selection(decision)
    except ValueError:
        known_capabilities, unknown_capabilities, inspected_tools = (
            inspect_capability_selection(decision)
        )
        inspected_skills = []

    try:
        with console.status("初始化 Agent..."):
            _, selected_skills, selected_tools = prepare_agent_for_message(
                cfg=cfg,
                state=state,
                decision=decision,
            )
    except Exception as e:
        console.print("[red]Agent 初始化失败:[/red]", end=" ")
        console.print(str(e), markup=False)
        return False

    console.print(
        f"[dim]route={decision.intent}, "
        f"capabilities={decision.required_capabilities}, "
        f"known={known_capabilities}, "
        f"unknown={unknown_capabilities}, "
        f"skills={selected_skills}, "
        f"tools={selected_tools}, "
        f"confidence={decision.confidence:.2f}[/dim]"
    )

    if inspected_tools != selected_tools:
        console.print(
            f"[yellow]Tool selection mismatch: "
            f"inspected={inspected_tools}, selected={selected_tools}[/yellow]"
        )

    if inspected_skills and inspected_skills != selected_skills:
        console.print(
            f"[yellow]Skill selection mismatch: "
            f"inspected={inspected_skills}, selected={selected_skills}[/yellow]"
        )

    try:
        printed_tool_calls: set[str] = set()

        for msg in stream_agent_response(
            cfg,
            state,
            user_input,
            isolated_messages=isolated_messages,
        ):
            _render_message(msg, printed_tool_calls)

    except Exception as e:
        console.print("[red]Agent 出错:[/red]", end=" ")
        console.print(str(e), markup=False)
        return False

    return True

# ---------------------------------------------------------------------------
# Plan rendering / execution input
# ---------------------------------------------------------------------------


def _render_plan_summary(plan: TaskPlan) -> None:
    """渲染计划摘要。"""

    console.print()
    console.print(
        f"Plan Mode: {plan.mode} | {plan.interpretation} | {plan.target_scope}",
        style="bold cyan",
    )

    if plan.objective:
        console.print(f"目标: {plan.objective}", style="cyan", markup=False)

    console.print(f"步骤数: {len(plan.steps)}", style="cyan")

    for index, step in enumerate(plan.steps, start=1):
        marker = "?" if step.requires_user_input else "✓"
        console.print(
            f"{index}. [{marker}] {step.instruction}",
            style="cyan",
            markup=False,
        )


def build_step_execution_input(step: PlanStep) -> str:
    """把 PlanStep 转成可交给 Router/Agent 的单步请求。"""

    input_lines: list[str] = []

    for item in step.inputs:
        ref = item.ref or "(无引用)"
        role = f"\n  用途：{item.role}" if item.role else ""
        input_lines.append(f"- {item.kind}: {ref}{role}")

    expected_lines = [f"- {item}" for item in step.expected_outputs]

    inputs_block = "\n".join(input_lines) if input_lines else "无显式输入引用。"
    expected_block = "\n".join(expected_lines) if expected_lines else "完成该步骤。"

    return f"""当前计划步骤：{step.id}

任务：
{step.instruction}

目的：
{step.purpose or "完成该计划步骤。"}

本步骤输入引用：
{inputs_block}

期望产出：
{expected_block}

执行要求：
- 只完成当前计划步骤，不要跳到后续步骤。
- 如果本步骤需要查看文件内容、预览数据、统计数据或验证数据，必须调用合适工具完成。
- 不要仅凭历史对话、当前任务状态或记忆声称已经完成预览、统计、分析或检查。
- 当前任务状态只能作为上下文线索，不等同于本步骤的工具结果。
- 如果本步骤需要预览文件，回答中必须基于本步骤实际工具返回的内容。
- 不要在步骤完成后询问“是否需要继续”，后续步骤由 Runtime 控制。

请完成这个计划步骤。"""


# ---------------------------------------------------------------------------
# Tool call rendering
# ---------------------------------------------------------------------------


def _render_tool_result(content: str) -> None:
    """渲染工具执行结果：错误显示关键行，成功输出显示前几行预览。"""

    if not content:
        console.print("[green]✓[/green] [dim](无输出)[/dim]")
        return

    size = len(content)
    lines = content.splitlines()

    is_error = (
        content.startswith("执行错误:")
        or "Traceback" in content[:200]
        or any(k in content for k in ("Error:", "Exception:"))
    )

    if is_error:
        last = next((ln.strip() for ln in reversed(lines) if ln.strip()), "")
        console.print("[red]✗[/red]", end=" ")
        console.print(_truncate(last, 120), style="dim", markup=False)
        return

    if size <= 200 and len(lines) <= 1:
        console.print("[green]✓[/green]", end=" ")
        console.print(content.strip(), style="dim", markup=False)
        return

    preview_lines = lines[:5]
    console.print("[green]✓[/green]")

    for line in preview_lines:
        console.print(f"  {_truncate(line, 100)}", style="dim", markup=False)

    if len(lines) > 5:
        console.print(
            f"  ... ({len(lines) - 5} 行未显示, 共 {size} 字符)",
            style="dim",
            markup=False,
        )


def _render_message(msg, printed_tool_calls: set) -> None:
    """根据消息类型渲染 tool call、tool result、最终回答。"""

    if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
        for tool_call in msg.tool_calls:
            tool_call_id = tool_call.get("id", "")

            if tool_call_id in printed_tool_calls:
                continue

            printed_tool_calls.add(tool_call_id)

            name = tool_call.get("name", "?")
            args = tool_call.get("args", {}) or {}

            console.print(f"\n[cyan]⚙ {name}[/cyan]")
            _print_tool_args(args)

        return

    if isinstance(msg, ToolMessage):
        content = str(msg.content) if msg.content else ""
        _render_tool_result(content)
        return

    if isinstance(msg, AIMessage):
        content = msg.content
        if content:
            text = content if isinstance(content, str) else str(content)
            console.print()
            console.print(text, style="bold green", markup=False)


def _print_tool_args(args: dict) -> None:
    """打印 tool 参数：单个短参数显示完整，否则只显示键名。"""

    if not args:
        return

    if "code" in args and len(args) == 1:
        code = str(args["code"])
        first_line = code.split("\n")[0][:80]
        more = " ..." if "\n" in code or len(code) > 80 else ""
        console.print(f"  {first_line}{more}", style="dim", markup=False)
        return

    for key, value in args.items():
        value_text = str(value)

        if len(value_text) > 60:
            value_text = value_text[:60] + "..."

        console.print(f"  {key}={value_text}", style="dim", markup=False)


def _truncate(text: str, max_len: int) -> str:
    """单行截断，超过 max_len 用 ... 收尾。"""

    if len(text) <= max_len:
        return text

    return text[: max_len - 3] + "..."


# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------


def _show_help() -> None:
    """显示 REPL 帮助信息。"""

    console.print("[bold]可用命令：[/bold]")
    console.print("  /help            显示帮助")
    console.print("  /config          配置 LLM")
    console.print("  /langsmith       配置 LangSmith 追踪（可选）")
    console.print("  /reset           清空对话历史、任务状态与 Python 命名空间")
    console.print("  /state           查看当前任务状态")
    console.print("  /route 文本      调试 Intent Router")
    console.print("  /decompose 文本  调试多任务拆解")
    console.print("  /plan 文本       调试任务规划")
    console.print("  /plan-mode       查看 Plan Mode 状态")
    console.print("  /plan-mode on    开启 Plan Mode")
    console.print("  /plan-mode off   关闭 Plan Mode")
    console.print("  /quit            退出")
    console.print("  其他输入会发送给 Agent")


# ---------------------------------------------------------------------------
# /config
# ---------------------------------------------------------------------------


def _do_config(session: PromptSession) -> None:
    """交互式配置 LLM provider / model / api key。"""

    current = load_config()

    console.print("[bold cyan]LLM 配置[/bold cyan]")

    if current:
        console.print(
            f"[dim]当前: {PROVIDER_LABELS[current.provider]} · "
            f"{current.model} · {current.masked_key()}[/dim]"
        )

    console.print("[dim]随时按 Ctrl+C 取消[/dim]")

    try:
        provider = _pick_provider(session, current)
        model = _pick_model(session, provider, current)
        api_key = _input_api_key(session, provider, current)
    except _Cancelled:
        console.print("\n[yellow]已取消，未保存任何变更。[/yellow]")
        return

    config = LLMConfig(provider=provider, model=model, api_key=api_key)
    save_config(config)

    console.print(
        f"\n[green]✓ 已保存[/green] "
        f"{PROVIDER_LABELS[config.provider]} · "
        f"{config.model} · "
        f"{config.masked_key()}"
    )
    console.print(f"[dim]配置文件: {CONFIG_FILE}[/dim]")


# ---------------------------------------------------------------------------
# /langsmith
# ---------------------------------------------------------------------------


def _do_langsmith(session: PromptSession) -> None:
    """交互式配置 LangSmith。"""

    current = load_langsmith_config()

    console.print("[bold cyan]LangSmith 追踪配置[/bold cyan]")
    console.print(
        "[dim]LangSmith 用于在网页端可视化 Agent 的每次推理与 tool call。[/dim]"
    )
    console.print("[dim]注册地址: https://smith.langchain.com/[/dim]")

    if current and current.enabled and current.api_key:
        console.print(
            f"[dim]当前: 已启用 · project={current.project} · "
            f"{current.masked_key()}[/dim]"
        )
    else:
        console.print("[dim]当前: 未启用[/dim]")

    console.print("[dim]按 Ctrl+C 取消[/dim]")

    console.print("\n[bold]操作:[/bold]")
    console.print("  [cyan]1.[/cyan] 配置 / 更新 API Key")
    console.print("  [cyan]2.[/cyan] 关闭 LangSmith")
    console.print("  [cyan]3.[/cyan] 取消")

    try:
        while True:
            raw = session.prompt(HTML("<ansigreen>› </ansigreen>")).strip()

            if raw == "1":
                _langsmith_set(session, current)
                return

            if raw == "2":
                clear_langsmith_config()
                console.print("[green]✓ 已关闭 LangSmith[/green]")
                return

            if raw == "3" or not raw:
                console.print("[yellow]已取消[/yellow]")
                return

            console.print("[red]请输入 1 / 2 / 3[/red]")
    except (KeyboardInterrupt, EOFError):
        console.print("\n[yellow]已取消[/yellow]")


def _langsmith_set(session: PromptSession, current) -> None:
    """配置 LangSmith API key 与 project。"""

    from prompt_toolkit import prompt as pt_prompt

    existing_key = current.api_key if current else ""

    if existing_key:
        masked = (
            f"{existing_key[:4]}...{existing_key[-4:]}"
            if len(existing_key) > 8
            else "*" * len(existing_key)
        )
        console.print(
            f"\n[bold]LangSmith API Key[/bold] "
            f"[dim](Enter 保留: {masked})[/dim]"
        )
    else:
        console.print("\n[bold]LangSmith API Key[/bold]")

    try:
        key = pt_prompt(HTML("<ansigreen>› </ansigreen>"), is_password=True).strip()
    except (KeyboardInterrupt, EOFError):
        console.print("\n[yellow]已取消[/yellow]")
        return

    if not key and existing_key:
        key = existing_key

    if not key:
        console.print("[red]API Key 不可空白[/red]")
        return

    default_project = current.project if current else "scarecrow"
    console.print(
        f"\n[bold]Project 名称[/bold] "
        f"[dim](Enter 使用: {default_project})[/dim]"
    )

    try:
        project = pt_prompt(HTML("<ansigreen>› </ansigreen>")).strip()
    except (KeyboardInterrupt, EOFError):
        console.print("\n[yellow]已取消[/yellow]")
        return

    if not project:
        project = default_project

    save_langsmith_config(
        LangSmithConfig(api_key=key, project=project, enabled=True)
    )

    masked_new = f"{key[:4]}...{key[-4:]}" if len(key) > 8 else "*" * len(key)

    console.print(
        f"\n[green]✓ 已保存[/green] LangSmith · "
        f"project={project} · {masked_new}"
    )


# ---------------------------------------------------------------------------
# Provider / Model / Key prompts
# ---------------------------------------------------------------------------


def _pick_provider(session: PromptSession, current) -> str:
    """选择 provider。"""

    providers = list(PROVIDER_MODELS.keys())

    console.print("\n[bold]选择 Provider:[/bold]")

    for index, provider in enumerate(providers, start=1):
        console.print(f"  [cyan]{index}.[/cyan] {PROVIDER_LABELS[provider]}")

    default_index = None

    if current:
        for index, provider in enumerate(providers, start=1):
            if provider == current.provider:
                default_index = index
                break

    hint = f"数字 (1-{len(providers)})"

    if default_index:
        hint += f"，默认 {default_index}"

    while True:
        try:
            raw = session.prompt(HTML("<ansigreen>› </ansigreen>")).strip()
        except (KeyboardInterrupt, EOFError):
            raise _Cancelled()

        if not raw and default_index:
            return providers[default_index - 1]

        if raw.isdigit() and 1 <= int(raw) <= len(providers):
            return providers[int(raw) - 1]

        console.print(f"[red]请输入 {hint}[/red]")


def _pick_model(session: PromptSession, provider: str, current) -> str:
    """选择 model。"""

    models = PROVIDER_MODELS.get(provider, [])

    console.print(f"\n[bold]选择 Model ({PROVIDER_LABELS[provider]}):[/bold]")

    for index, model in enumerate(models, start=1):
        console.print(f"  [cyan]{index}.[/cyan] {model}")

    console.print("  [cyan]c.[/cyan] 自定义模型名")

    default_index = None

    if current and current.provider == provider:
        for index, model in enumerate(models, start=1):
            if model == current.model:
                default_index = index
                break

    hint = f"数字 (1-{len(models)}) 或 c"

    if default_index:
        hint += f"，默认 {default_index}"

    while True:
        try:
            raw = session.prompt(HTML("<ansigreen>› </ansigreen>")).strip()
        except (KeyboardInterrupt, EOFError):
            raise _Cancelled()

        if not raw and default_index:
            return models[default_index - 1]

        if raw.lower() == "c":
            model = session.prompt(
                HTML("<ansigreen>模型名 › </ansigreen>")
            ).strip()

            if model:
                return model

            console.print("[red]模型名不可空白[/red]")
            continue

        if raw.isdigit() and 1 <= int(raw) <= len(models):
            return models[int(raw) - 1]

        console.print(f"[red]请输入 {hint}[/red]")


def _input_api_key(session: PromptSession, provider: str, current) -> str:
    """输入 API key。"""

    from prompt_toolkit import prompt as pt_prompt

    existing_key = (
        current.api_key
        if current and current.provider == provider and current.api_key
        else ""
    )

    if existing_key:
        masked = (
            f"{existing_key[:4]}...{existing_key[-4:]}"
            if len(existing_key) > 8
            else "*" * len(existing_key)
        )
        console.print(
            f"\n[bold]API Key[/bold] "
            f"[dim](Enter 保留: {masked})[/dim]"
        )
    else:
        console.print("\n[bold]API Key[/bold]")

    while True:
        try:
            key = pt_prompt(
                HTML("<ansigreen>› </ansigreen>"),
                is_password=True,
            ).strip()
        except (KeyboardInterrupt, EOFError):
            raise _Cancelled()

        if not key and existing_key:
            return existing_key

        if key:
            return key

        console.print("[red]API Key 不可空白[/red]")