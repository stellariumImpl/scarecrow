# src/scarecrow/tools/workspace.py

from pathlib import Path

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from scarecrow.workspace import (
    resolve_workspace_path,
    scan_data_files,
)


_WORKSPACE: Path | None = None

_IGNORED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    "node_modules",
    ".idea",
    ".vscode",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "dist",
    "build",
    ".DS_Store",
}

_IGNORED_SUFFIXES = {
    ".pyc",
    ".pyo",
}


def set_workspace(workspace: Path) -> None:
    """设置当前工具可访问的 workspace。"""

    global _WORKSPACE
    _WORKSPACE = workspace


class ListDataFilesInput(BaseModel):
    """list_data_files 工具输入。"""

    limit: int = Field(
        default=50,
        ge=1,
        le=200,
        description="最多返回多少个数据文件路径。",
    )


@tool(args_schema=ListDataFilesInput)
def list_data_files(limit: int = 50) -> str:
    """列出当前工作区下的数据文件。

    只返回相对路径，不读取文件内容，不修改文件。
    """

    if _WORKSPACE is None:
        return "当前 workspace 尚未初始化。"

    files = scan_data_files(_WORKSPACE)

    if not files:
        return "当前工作区下未发现数据文件。"

    shown = files[:limit]
    lines = [f"- {path}" for path in shown]

    if len(files) > limit:
        lines.append(f"... 还有 {len(files) - limit} 个文件未显示")

    return "\n".join(lines)


class ListWorkspaceFilesInput(BaseModel):
    """list_workspace_files 工具输入。"""

    max_depth: int = Field(
        default=3,
        ge=1,
        le=8,
        description="递归展示目录结构的最大深度。",
    )

    limit: int = Field(
        default=120,
        ge=1,
        le=500,
        description="最多返回多少个文件或目录条目。",
    )


@tool(args_schema=ListWorkspaceFilesInput)
def list_workspace_files(max_depth: int = 3, limit: int = 120) -> str:
    """列出当前工作区的项目结构。

    只展示目录和文件路径，不读取文件内容，不修改文件。
    默认会跳过 .git、虚拟环境、缓存目录、node_modules 等噪声目录。
    """

    if _WORKSPACE is None:
        return "当前 workspace 尚未初始化。"

    if not _WORKSPACE.exists() or not _WORKSPACE.is_dir():
        return f"当前 workspace 不存在或不是目录: {_WORKSPACE}"

    entries: list[str] = []

    _walk_workspace(
        root=_WORKSPACE,
        current=_WORKSPACE,
        max_depth=max_depth,
        limit=limit,
        entries=entries,
    )

    if not entries:
        return "当前工作区下未发现可展示的文件或目录。"

    lines = [f"Workspace: {_WORKSPACE}", ""]

    for item in entries[:limit]:
        lines.append(item)

    if len(entries) > limit:
        lines.append(f"... 还有 {len(entries) - limit} 个条目未显示")

    return "\n".join(lines)


class ResolveWorkspaceFileInput(BaseModel):
    """resolve_workspace_file 工具输入。"""

    query: str = Field(
        ...,
        description="用户提到的文件名、相对路径或文件名片段。",
    )

    data_files_only: bool = Field(
        default=False,
        description="是否只在数据文件中匹配。",
    )


@tool(args_schema=ResolveWorkspaceFileInput)
def resolve_workspace_file(query: str, data_files_only: bool = False) -> str:
    """解析用户提到的文件名或路径，返回工作区内最可能匹配的相对路径。

    只做路径匹配，不读取文件内容，不修改文件。
    """

    if _WORKSPACE is None:
        return "当前 workspace 尚未初始化。"

    matches = resolve_workspace_path(
        workspace=_WORKSPACE,
        query=query,
        data_files_only=data_files_only,
        max_candidates=10,
    )

    if not matches:
        return f"没有找到与 `{query}` 匹配的工作区文件。"

    if len(matches) == 1:
        return f"匹配到 1 个文件:\n- {matches[0]}"

    lines = [f"匹配到 {len(matches)} 个候选文件:"]
    lines.extend(f"- {path}" for path in matches)
    return "\n".join(lines)


def _walk_workspace(
    root: Path,
    current: Path,
    max_depth: int,
    limit: int,
    entries: list[str],
) -> None:
    """递归遍历 workspace，生成类 tree 的文本结构。"""

    if len(entries) >= limit:
        return

    try:
        children = sorted(
            current.iterdir(),
            key=lambda p: (not p.is_dir(), p.name.lower()),
        )
    except (PermissionError, OSError):
        return

    for child in children:
        if len(entries) >= limit:
            return

        if _should_ignore(child):
            continue

        try:
            rel = child.relative_to(root)
        except ValueError:
            continue

        depth = len(rel.parts)

        if depth > max_depth:
            continue

        indent = "  " * (depth - 1)
        suffix = "/" if child.is_dir() else ""
        entries.append(f"{indent}- {child.name}{suffix}")

        if child.is_dir():
            _walk_workspace(
                root=root,
                current=child,
                max_depth=max_depth,
                limit=limit,
                entries=entries,
            )


def _should_ignore(path: Path) -> bool:
    """判断路径是否应该在项目结构展示中跳过。"""

    if path.name.startswith("."):
        return True

    if path.name in _IGNORED_DIRS:
        return True

    if path.suffix in _IGNORED_SUFFIXES:
        return True

    return False