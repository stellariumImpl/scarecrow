# src/scarecrow/tools/workspace.py

from pathlib import Path

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from scarecrow.workspace import scan_data_files


_WORKSPACE: Path | None = None


def set_workspace(workspace: Path) -> None:
    """设置当前工具可访问的 workspace。

    这是本地 CLI 工具，workspace 来自 scarecrow 启动目录。
    """

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