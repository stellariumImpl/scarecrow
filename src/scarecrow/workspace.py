# src/scarecrow/workspace.py

"""工作区扫描:列出数据文件,供 system prompt 使用"""

from pathlib import Path

# 视为"数据文件"的扩展名
_DATA_EXTS = {".csv", ".parquet", ".jsonl", ".xlsx", ".xls"}

# 扫描时跳过的目录名
_IGNORED_DIRS = {
    ".git", ".venv", "venv", "env", "__pycache__",
    "node_modules", ".idea", ".vscode", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", "dist", "build", ".DS_Store",
}

# 最多列出的文件数,防止上下文爆炸
_MAX_FILES = 50


def scan_data_files(workspace: Path) -> list[str]:
    """递归扫描工作区,返回相对路径形式的数据文件列表"""
    if not workspace.exists() or not workspace.is_dir():
        return []

    found: list[str] = []
    for path in _walk(workspace):
        if path.suffix.lower() not in _DATA_EXTS:
            continue
        try:
            rel = path.relative_to(workspace)
        except ValueError:
            continue
        found.append(str(rel))
        if len(found) >= _MAX_FILES:
            break

    found.sort()
    return found


def _walk(root: Path):
    """手写递归遍历,自动跳过 ignored 目录(rglob 没法跳目录)"""
    try:
        entries = list(root.iterdir())
    except (PermissionError, OSError):
        return

    for entry in entries:
        if entry.name in _IGNORED_DIRS or entry.name.startswith("."):
            # 隐藏文件/目录都跳过
            continue
        if entry.is_dir():
            yield from _walk(entry)
        elif entry.is_file():
            yield entry


def workspace_brief(workspace: Path) -> str:
    """生成可注入 system prompt 的工作区简报"""
    files = scan_data_files(workspace)
    header = f"## 当前工作区\n\n路径: `{workspace}`"

    if not files:
        return f"{header}\n\n工作区下未发现数据文件 (csv / parquet / jsonl / xlsx)。"

    truncated = len(files) >= _MAX_FILES
    file_lines = "\n".join(f"- `{f}`" for f in files)
    note = f"\n\n(已截断,只显示前 {_MAX_FILES} 个)" if truncated else ""

    return (
        f"{header}\n\n"
        f"数据文件清单 (相对工作区路径,共 {len(files)} 个):\n{file_lines}"
        f"{note}\n\n"
        f"用户提到文件名时,优先匹配以上清单。"
        f"用户没明说路径时,直接使用清单中的相对路径,不要再 `os.listdir` 试探。"
    )