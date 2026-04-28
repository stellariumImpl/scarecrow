# src/scarecrow/tools.py

"""Agent 工具集"""

import contextlib
import io
import traceback

from langchain_core.tools import tool
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 持久 Python 命名空间：跨多次工具调用保留变量
# ---------------------------------------------------------------------------
_NAMESPACE: dict = {}


def reset_namespace() -> None:
    """清空 Python 命名空间"""
    _NAMESPACE.clear()


# ---------------------------------------------------------------------------
# Tool: run_python
# ---------------------------------------------------------------------------
class RunPythonInput(BaseModel):
    """run_python 工具的输入参数"""

    code: str = Field(
        ...,
        description=(
            "要执行的 Python 代码。"
            "环境已预装 pandas (as pd) 与 numpy (as np)。"
            "想看到输出必须用 print()，函数返回值不会自动显示。"
        ),
    )


@tool(args_schema=RunPythonInput)
def run_python(code: str) -> str:
    """在持久 Python 环境中执行代码，用于数据分析。

    详细约定与适用场景见 SKILL.md（run-python）。
    """
    # 首次调用注入常用 import
    if "pd" not in _NAMESPACE:
        try:
            import numpy as np
            import pandas as pd
            _NAMESPACE["pd"] = pd
            _NAMESPACE["np"] = np
        except ImportError as e:
            return f"环境缺失依赖: {e}。请安装 pandas 和 numpy。"

    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            exec(code, _NAMESPACE)
        out = buf.getvalue()
        return out if out else "(执行成功，无输出)"
    except Exception:
        return f"执行错误:\n{traceback.format_exc()}"