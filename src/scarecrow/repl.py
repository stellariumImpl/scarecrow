"""Scarecrow REPL - 交互式对话循环"""

from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from rich.console import Console

from scarecrow.config import (
    PROVIDER_LABELS,
    PROVIDER_MODELS,
    LLMConfig,
    load_config,
    save_config,
    CONFIG_FILE,
    CONFIG_DIR,
)
from scarecrow.workspace import workspace_summary

console = Console()


class _Cancelled(Exception):
    """用户取消配置流程"""


def start_repl(workspace: Path) -> None:
    """启动 REPL 循环"""
    history_file = CONFIG_DIR / "history.txt"
    session: PromptSession = PromptSession(history=FileHistory(str(history_file)))

    console.print(f"Scarecrow — 当前工作区: {workspace}")
    console.print(workspace_summary(workspace))
    console.print("\n输入 /help 查看命令，/quit 退出")

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
        else:
            console.print(f"收到: {user_input}")


def _show_help() -> None:
    console.print("[bold]可用命令：[/bold]")
    console.print("  /help    显示帮助")
    console.print("  /config  配置 LLM")
    console.print("  /quit    退出")
    console.print("  其他输入会发送给 Agent（尚未实现）")


def _do_config(session: PromptSession) -> None:
    """交互式配置 LLM（菜单风格）"""
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
            raw = session.prompt(HTML(f"<ansigreen>› </ansigreen>")).strip()
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
            return session.prompt(
                HTML("<ansigreen>模型名称 › </ansigreen>")
            ).strip()
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
            raw = session.prompt(HTML(f"<ansigreen>› </ansigreen>")).strip()
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
    label = PROVIDER_LABELS.get(provider, provider)
    existing = None
    if current and current.provider == provider:
        existing = current.api_key

    if existing:
        masked = f"{existing[:4]}...{existing[-4:]}" if len(existing) > 8 else "*" * len(existing)
        console.print(
            f"\n[bold]输入 {label} API Key[/bold] "
            f"[dim](Enter 保留当前: {masked})[/dim]"
        )
    else:
        console.print(f"\n[bold]输入 {label} API Key[/bold]")

    try:
        key = session.prompt(
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
