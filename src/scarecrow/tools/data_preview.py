# src/scarecrow/tools/data_preview.py

from pathlib import Path

from langchain_core.tools import tool
from pydantic import BaseModel, Field


_WORKSPACE: Path | None = None

_SUPPORTED_EXTS = {
    ".csv",
    ".parquet",
    ".jsonl",
    ".xlsx",
    ".xls",
}


def set_preview_workspace(workspace: Path) -> None:
    """设置 preview_data_file 可访问的 workspace。"""

    global _WORKSPACE
    _WORKSPACE = workspace


class PreviewDataFileInput(BaseModel):
    """preview_data_file 工具输入。"""

    path: str = Field(
        ...,
        description="工作区内的数据文件相对路径，例如 data/users.csv。",
    )

    rows: int = Field(
        default=5,
        ge=1,
        le=50,
        description="预览多少行，默认 5 行。",
    )


@tool(args_schema=PreviewDataFileInput)
def preview_data_file(path: str, rows: int = 5) -> str:
    """安全预览工作区内数据文件的前几行。

    支持 csv、parquet、jsonl、xlsx、xls。
    只读取少量行，不修改文件。
    """

    if _WORKSPACE is None:
        return "当前 workspace 尚未初始化。"

    target = _safe_resolve_workspace_path(_WORKSPACE, path)

    if target is None:
        return f"路径不合法或不在当前 workspace 内: {path}"

    if not target.exists() or not target.is_file():
        return f"文件不存在: {path}"

    if target.suffix.lower() not in _SUPPORTED_EXTS:
        return (
            f"不支持的文件类型: {target.suffix}。"
            "当前仅支持 csv / parquet / jsonl / xlsx / xls。"
        )

    try:
        return _preview_with_pandas(target, rows)
    except ImportError:
        return "环境缺失 pandas，无法预览数据文件。"
    except Exception as e:
        return f"预览失败: {type(e).__name__}: {e}"


def _safe_resolve_workspace_path(workspace: Path, relative_path: str) -> Path | None:
    """安全解析 workspace 内相对路径，防止路径逃逸。"""

    try:
        workspace_resolved = workspace.resolve()
        target = (workspace_resolved / relative_path).resolve()
    except OSError:
        return None

    if target == workspace_resolved:
        return None

    if workspace_resolved not in target.parents:
        return None

    return target


def _preview_with_pandas(path: Path, rows: int) -> str:
    """用 pandas 读取少量数据并返回文本预览。"""

    import pandas as pd

    suffix = path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(path, nrows=rows)
    elif suffix == ".jsonl":
        df = pd.read_json(path, lines=True, nrows=rows)
    elif suffix == ".parquet":
        df = pd.read_parquet(path)
        df = df.head(rows)
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path, nrows=rows)
    else:
        return f"不支持的文件类型: {suffix}"

    lines = [
        f"文件: {path.name}",
        f"路径: {path}",
        f"预览行数: {min(len(df), rows)}",
        f"列数: {len(df.columns)}",
        "",
        "列名:",
        ", ".join(str(col) for col in df.columns),
        "",
        "前几行:",
        df.to_string(index=False),
    ]

    return "\n".join(lines)