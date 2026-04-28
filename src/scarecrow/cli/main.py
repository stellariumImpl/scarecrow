# src/scarecrow/cli/main.py

from pathlib import Path

import typer

from scarecrow.repl import start_repl

app = typer.Typer(help="Scarecrow - AI 数据分析助手")


@app.callback(invoke_without_command=True)
def main() -> None:
    """启动 Scarecrow REPL"""
    start_repl(Path.cwd())


if __name__ == "__main__":
    app()
