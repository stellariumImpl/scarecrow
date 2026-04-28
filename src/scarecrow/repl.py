# src/scarecrow/repl.py

"""Scarecrow REPL - 交互式对话循环"""

from pathlib import Path

from langchain.agents import create_agent
from scarecrow.llm.loader import load_chat_model_from_config
from langchain_core.messages import AIMessage, ToolMessage
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from rich.console import Console

from scarecrow.config import (
    PROVIDER_LABELS,
    PROVIDER_MODELS,
    LLMConfig,
    LangSmithConfig,
    CONFIG_FILE,
    CONFIG_DIR,
    SKILLS_DIR,
    clear_langsmith_config,
    ensure_config_dir,
    load_config,
    load_langsmith_config,
    save_config,
    save_langsmith_config,
)
from scarecrow.langsmith_setup import apply_langsmith_env
from scarecrow.context import build_system_prompt
from scarecrow.skills import ensure_builtin_skills
from scarecrow.tools import run_python, reset_namespace

from scarecrow.router import IntentRouter


def _debug_route(text: str, agent, workspace: Path) -> None:
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
            model = load_chat_model_from_config(cfg)
            router = IntentRouter(model)
            decision = router.route(text)
    except Exception as e:
        console.print(f"[red]Router 出错: {e}[/red]")
        return

    console.print_json(decision.model_dump_json(indent=2, ensure_ascii=False))


console = Console()


class _Cancelled(Exception):
    """用户取消配置流程"""


def start_repl(workspace: Path) -> None:
    """启动 REPL 循环"""
    ensure_config_dir()
    SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    ensure_builtin_skills(SKILLS_DIR)

    # 启动时根据配置开启 LangSmith
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

    agent = None
    messages: list = []

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
        elif user_input == "/help":
            _show_help()
        elif user_input == "/config":
            _do_config(session)
            agent = None
        elif user_input == "/langsmith":
            _do_langsmith(session)
            apply_langsmith_env()
            agent = None  # tracing 状态变了，重建 agent 让新设置生效
        elif user_input == "/reset":
            messages = []
            reset_namespace()
            console.print("[dim]已清空对话历史与 Python 命名空间[/dim]")
        elif user_input.startswith("/route "):
            _debug_route(user_input.removeprefix("/route ").strip(), agent, workspace)
        else:
            agent, messages = _handle_chat(user_input, agent, messages, workspace)


# ---------------------------------------------------------------------------
# 对话处理 + tool call 可视化
# ---------------------------------------------------------------------------


def _handle_chat(user_input: str, agent, messages: list, workspace: Path):
    """处理一轮对话,流式打印 tool call 过程"""
    if agent is None:
        cfg = load_config()
        if cfg is None:
            console.print("[yellow]请先运行 /config 配置 LLM[/yellow]")
            return agent, messages
        try:
            with console.status("初始化 Agent..."):
                agent = _build_agent(cfg, workspace)  # ← 这里加 workspace
        except Exception as e:
            console.print(f"[red]Agent 初始化失败: {e}[/red]")
            return agent, messages

    messages.append({"role": "user", "content": user_input})

    try:
        # 用 stream 模式取每一步更新，便于实时打印 tool call
        result_messages = list(messages)
        printed_tool_calls: set[str] = set()

        for chunk in agent.stream({"messages": messages}, stream_mode="values"):
            new_messages = chunk.get("messages", [])
            # 只处理新增的消息
            for msg in new_messages[len(result_messages) :]:
                _render_message(msg, printed_tool_calls)
            result_messages = new_messages

        messages = result_messages
    except Exception as e:
        messages.pop()
        console.print(f"[red]Agent 出错: {e}[/red]")

    return agent, messages


def _render_tool_result(content: str) -> None:
    """渲染工具执行结果:错误显示关键行、成功输出显示前几行预览"""
    if not content:
        console.print("[green]✓[/green] [dim](无输出)[/dim]")
        return

    size = len(content)
    lines = content.splitlines()

    # 1. 错误优先识别:展示最有用的尾部
    is_error = (
        content.startswith("执行错误:")
        or "Traceback" in content[:200]
        or any(k in content for k in ("Error:", "Exception:"))
    )
    if is_error:
        # 错误的关键信息在最后一行(异常类型 + 消息)
        # 取最后一个非空行,如有 ":" 通常就是 ErrorType: message
        last = next((ln.strip() for ln in reversed(lines) if ln.strip()), "")
        console.print(f"[red]✗[/red] [dim]{_truncate(last, 120)}[/dim]")
        return

    # 2. 短输出(单行 + 不超 200 字符)直接显示
    if size <= 200 and len(lines) <= 1:
        console.print(f"[green]✓[/green] [dim]{content.strip()}[/dim]")
        return

    # 3. 长输出:显示前 5 行预览 + 总字符数
    preview_lines = lines[:5]
    preview = "\n".join(f"  [dim]{_truncate(ln, 100)}[/dim]" for ln in preview_lines)
    more = (
        f"\n  [dim]... ({len(lines) - 5} 行未显示, 共 {size} 字符)[/dim]"
        if len(lines) > 5
        else ""
    )
    console.print(f"[green]✓[/green]\n{preview}{more}")


def _truncate(s: str, max_len: int) -> str:
    """单行截断:超过 max_len 用 ... 收尾"""
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def _render_message(msg, printed_tool_calls: set) -> None:
    """根据消息类型渲染：tool call / tool result / 最终回答"""
    # 1. AI 决定调用工具
    if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
        for tc in msg.tool_calls:
            tc_id = tc.get("id", "")
            if tc_id in printed_tool_calls:
                continue
            printed_tool_calls.add(tc_id)
            name = tc.get("name", "?")
            args = tc.get("args", {}) or {}
            console.print(f"\n[cyan]⚙ {name}[/cyan]")
            _print_tool_args(args)
        return

    # 2. 工具执行结果
    if isinstance(msg, ToolMessage):
        content = str(msg.content) if msg.content else ""
        _render_tool_result(content)
        return

    # 3. AI 最终回答（无 tool_calls 的 AIMessage）
    if isinstance(msg, AIMessage):
        content = msg.content
        if content:
            text = content if isinstance(content, str) else str(content)
            console.print(f"\n[bold green]{text}[/bold green]")


def _print_tool_args(args: dict) -> None:
    """打印 tool 参数：单个短参数显示完整，否则只显示键名"""
    if not args:
        return
    # run_python 的 code 参数特殊处理：截断显示
    if "code" in args and len(args) == 1:
        code = str(args["code"])
        first_line = code.split("\n")[0][:80]
        more = " ..." if "\n" in code or len(code) > 80 else ""
        console.print(f"  [dim]{first_line}{more}[/dim]")
        return
    # 通用参数：键值对缩略
    for k, v in args.items():
        v_str = str(v)
        if len(v_str) > 60:
            v_str = v_str[:60] + "..."
        console.print(f"  [dim]{k}={v_str}[/dim]")


# ---------------------------------------------------------------------------
# Agent 构建
# ---------------------------------------------------------------------------


def _build_agent(cfg: LLMConfig, workspace: Path):
    """按配置和工作区构建 Agent"""
    model = load_chat_model_from_config(cfg)

    return create_agent(
        model=model,
        tools=[run_python],
        system_prompt=build_system_prompt(
            workspace=workspace,
            skills_dir=SKILLS_DIR,
        ),
    )


# ---------------------------------------------------------------------------
# 帮助
# ---------------------------------------------------------------------------


def _show_help() -> None:
    console.print("[bold]可用命令：[/bold]")
    console.print("  /help       显示帮助")
    console.print("  /config     配置 LLM")
    console.print("  /langsmith  配置 LangSmith 追踪（可选）")
    console.print("  /reset      清空对话历史与 Python 命名空间")
    console.print("  /route 文本  调试 Intent Router")
    console.print("  /quit       退出")
    console.print("  其他输入会发送给 Agent")


# ---------------------------------------------------------------------------
# /config 流程（不变，照旧）
# ---------------------------------------------------------------------------


def _do_config(session: PromptSession) -> None:
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
        f"{PROVIDER_LABELS[config.provider]} · {config.model} · {config.masked_key()}"
    )
    console.print(f"[dim]配置文件: {CONFIG_FILE}[/dim]")


# ---------------------------------------------------------------------------
# /langsmith 流程
# ---------------------------------------------------------------------------


def _do_langsmith(session: PromptSession) -> None:
    """交互式配置 LangSmith"""
    current = load_langsmith_config()

    console.print("[bold cyan]LangSmith 追踪配置[/bold cyan]")
    console.print(
        "[dim]LangSmith 用于在网页端可视化 Agent 的每次推理与 tool call。[/dim]"
    )
    console.print("[dim]注册地址: https://smith.langchain.com/[/dim]")
    if current and current.enabled and current.api_key:
        console.print(
            f"[dim]当前: 已启用 · project={current.project} · {current.masked_key()}[/dim]"
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
    """配置 LangSmith api key 与 project"""
    from prompt_toolkit import prompt as pt_prompt  # 临时一次性 prompt

    # API Key（用独立的 prompt，避免污染主 session 的密码状态）
    existing_key = current.api_key if current else ""
    if existing_key:
        masked = (
            f"{existing_key[:4]}...{existing_key[-4:]}"
            if len(existing_key) > 8
            else "*" * len(existing_key)
        )
        console.print(
            f"\n[bold]LangSmith API Key[/bold] [dim](Enter 保留: {masked})[/dim]"
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

    # Project 名称（普通输入，绝对不能加密码模式）
    default_project = current.project if current else "scarecrow"
    console.print(
        f"\n[bold]Project 名称[/bold] [dim](Enter 使用: {default_project})[/dim]"
    )
    try:
        project = pt_prompt(HTML("<ansigreen>› </ansigreen>")).strip()
    except (KeyboardInterrupt, EOFError):
        console.print("\n[yellow]已取消[/yellow]")
        return
    if not project:
        project = default_project

    save_langsmith_config(LangSmithConfig(api_key=key, project=project, enabled=True))
    masked_new = f"{key[:4]}...{key[-4:]}" if len(key) > 8 else "*" * len(key)
    console.print(
        f"\n[green]✓ 已保存[/green] LangSmith · project={project} · {masked_new}"
    )


# ---------------------------------------------------------------------------
# Provider / Model / Key 交互（与之前一致）
# ---------------------------------------------------------------------------


def _pick_provider(session: PromptSession, current) -> str:
    providers = list(PROVIDER_MODELS.keys())
    console.print("\n[bold]选择 Provider:[/bold]")
    for i, p in enumerate(providers, start=1):
        console.print(f"  [cyan]{i}.[/cyan] {PROVIDER_LABELS[p]}")

    default_idx = None
    if current:
        for i, p in enumerate(providers):
            if p == current.provider:
                default_idx = i + 1
                break

    hint = f"数字 (1-{len(providers)})"
    if default_idx:
        hint += f"，默认 {default_idx}"

    while True:
        try:
            raw = session.prompt(HTML("<ansigreen>› </ansigreen>")).strip()
        except (KeyboardInterrupt, EOFError):
            raise _Cancelled()

        if not raw and default_idx:
            return providers[default_idx - 1]
        if raw.isdigit() and 1 <= int(raw) <= len(providers):
            return providers[int(raw) - 1]
        console.print(f"[red]请输入 {hint}[/red]")


def _pick_model(session: PromptSession, provider: str, current) -> str:
    models = PROVIDER_MODELS.get(provider, [])

    if not models:
        try:
            return session.prompt(HTML("<ansigreen>模型名称 › </ansigreen>")).strip()
        except (KeyboardInterrupt, EOFError):
            raise _Cancelled()

    console.print(f"\n[bold]选择 {PROVIDER_LABELS[provider]} 模型:[/bold]")
    for i, m in enumerate(models, start=1):
        console.print(f"  [cyan]{i}.[/cyan] {m}")
    console.print(f"  [cyan]{len(models) + 1}.[/cyan] [dim]自定义（手动输入）[/dim]")

    default_idx = None
    if current and current.provider == provider:
        for i, m in enumerate(models):
            if m == current.model:
                default_idx = i + 1
                break

    hint = f"数字 (1-{len(models) + 1})"
    if default_idx:
        hint += f"，默认 {default_idx}"

    while True:
        try:
            raw = session.prompt(HTML("<ansigreen>› </ansigreen>")).strip()
        except (KeyboardInterrupt, EOFError):
            raise _Cancelled()

        if not raw and default_idx:
            return models[default_idx - 1]
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(models):
                return models[idx - 1]
            if idx == len(models) + 1:
                try:
                    custom = session.prompt(
                        HTML("<ansigreen>自定义模型 › </ansigreen>")
                    ).strip()
                except (KeyboardInterrupt, EOFError):
                    raise _Cancelled()
                if custom:
                    return custom
                console.print("[red]模型名称不可空白[/red]")
                continue
        console.print(f"[red]请输入 {hint}[/red]")


def _input_api_key(session: PromptSession, provider: str, current) -> str:
    from prompt_toolkit import prompt as pt_prompt

    label = PROVIDER_LABELS.get(provider, provider)
    existing = None
    if current and current.provider == provider:
        existing = current.api_key

    if existing:
        masked = (
            f"{existing[:4]}...{existing[-4:]}"
            if len(existing) > 8
            else "*" * len(existing)
        )
        console.print(
            f"\n[bold]输入 {label} API Key[/bold] "
            f"[dim](Enter 保留当前: {masked})[/dim]"
        )
    else:
        console.print(f"\n[bold]输入 {label} API Key[/bold]")

    try:
        key = pt_prompt(
            HTML("<ansigreen>› </ansigreen>"),
            is_password=True,
        ).strip()
    except (KeyboardInterrupt, EOFError):
        raise _Cancelled()

    if not key and existing:
        return existing
    if not key:
        console.print("[red]API Key 不可空白[/red]")
        raise _Cancelled()
    return key
