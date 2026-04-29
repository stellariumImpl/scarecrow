# src/scarecrow/tools/registry.py

from collections.abc import Callable
from dataclasses import dataclass

from langchain_core.tools import BaseTool

from scarecrow.tools.python import run_python
from scarecrow.tools.schemas import ToolMetadata
from scarecrow.tools.workspace import list_data_files


@dataclass(frozen=True)
class ToolEntry:
    metadata: ToolMetadata
    tool: BaseTool | Callable


class ToolRegistry:
    """Tool 注册表。

    负责集中管理工具，避免在 repl.py / runtime 里硬编码 tools=[...]
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolEntry] = {}

    def register(self, entry: ToolEntry) -> None:
        self._tools[entry.metadata.name] = entry

    def get(self, name: str) -> ToolEntry | None:
        return self._tools.get(name)

    def list_metadata(self) -> list[ToolMetadata]:
        return [entry.metadata for entry in self._tools.values()]

    def select_tools(self, names: list[str]) -> list[BaseTool | Callable]:
        """根据工具名选择启用工具。

        不存在的工具会被忽略。
        """

        selected: list[BaseTool | Callable] = []

        for name in names:
            entry = self.get(name)
            if entry is None:
                continue
            if not entry.metadata.enabled:
                continue
            selected.append(entry.tool)

        return selected


def build_default_tool_registry() -> ToolRegistry:
    """构建默认 Tool Registry。"""

    registry = ToolRegistry()

    registry.register(
        ToolEntry(
            metadata=ToolMetadata(
                name="run_python",
                description="在持久 Python 命名空间中执行代码，用于本地数据分析。",
                risk_level="medium",
                enabled=True,
            ),
            tool=run_python,
        )
    )

    registry.register(
        ToolEntry(
            metadata=ToolMetadata(
                name="list_data_files",
                description="列出当前工作区下的数据文件，只读，不读取文件内容。",
                risk_level="low",
                enabled=True,
            ),
            tool=list_data_files,
        )
    )

    return registry