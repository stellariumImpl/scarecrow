# src/scarecrow/tools/registry.py

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from langchain_core.tools import BaseTool

from scarecrow.tools.data_preview import preview_data_file
from scarecrow.tools.python import run_python
from scarecrow.tools.schemas import ToolMetadata, ToolRiskLevel
from scarecrow.tools.workspace import (
    list_data_files,
    list_workspace_files,
    resolve_workspace_file,
)


_RISK_ORDER: dict[ToolRiskLevel, int] = {
    "low": 0,
    "medium": 1,
    "high": 2,
}


@dataclass(frozen=True)
class ToolEntry:
    metadata: ToolMetadata
    tool: BaseTool | Callable


class ToolRegistry:
    """Tool 注册表。

    负责集中管理工具，并根据 capability 自动选择工具。
    Runtime 不应该写死 “某 intent 使用某 tool”。
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolEntry] = {}

    def register(self, entry: ToolEntry) -> None:
        self._tools[entry.metadata.name] = entry

    def get(self, name: str) -> ToolEntry | None:
        return self._tools.get(name)

    def list_metadata(self) -> list[ToolMetadata]:
        return [entry.metadata for entry in self._tools.values()]

    def list_capabilities(self) -> list[str]:
        """列出所有可用 capability。"""

        capabilities: set[str] = set()

        for entry in self._tools.values():
            if not entry.metadata.enabled:
                continue
            capabilities.update(entry.metadata.capabilities)

        return sorted(capabilities)

    def build_capability_index(self) -> str:
        """生成给 Router 使用的能力索引文本。"""

        lines = ["## 可用工具能力"]

        for entry in self._tools.values():
            meta = entry.metadata
            if not meta.enabled:
                continue

            lines.append("")
            lines.append(f"- 工具: `{meta.name}`")
            lines.append(f"  描述: {meta.description}")
            lines.append(f"  风险: {meta.risk_level}")
            lines.append(f"  能力: {', '.join(meta.capabilities) if meta.capabilities else '(无)'}")

        return "\n".join(lines)

    def select_tools(self, names: list[str]) -> list[BaseTool | Callable]:
        """根据工具名选择启用工具。

        这个方法保留给调试和兼容逻辑使用。
        新逻辑优先使用 select_tool_names_by_capabilities。
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

    def select_tool_names_by_capabilities(
        self,
        required_capabilities: list[str],
        max_risk: ToolRiskLevel = "medium",
    ) -> list[str]:
        """根据 required_capabilities 选择工具名。

        选择规则：
        - 工具必须 enabled
        - 工具风险不能超过 max_risk
        - 工具只要覆盖任一 required capability 就可以被选中
        - 返回顺序按注册顺序保持稳定
        """

        if not required_capabilities:
            return []

        required = set(required_capabilities)
        selected: list[str] = []

        for entry in self._tools.values():
            meta = entry.metadata

            if not meta.enabled:
                continue

            if not _risk_allowed(meta.risk_level, max_risk):
                continue

            if required.intersection(meta.capabilities):
                selected.append(meta.name)

        return selected

    def select_tools_by_capabilities(
        self,
        required_capabilities: list[str],
        max_risk: ToolRiskLevel = "medium",
    ) -> list[BaseTool | Callable]:
        """根据 capability 直接选择工具对象。"""

        names = self.select_tool_names_by_capabilities(
            required_capabilities=required_capabilities,
            max_risk=max_risk,
        )
        return self.select_tools(names)


def _risk_allowed(tool_risk: ToolRiskLevel, max_risk: ToolRiskLevel) -> bool:
    return _RISK_ORDER[tool_risk] <= _RISK_ORDER[max_risk]


def build_default_tool_registry() -> ToolRegistry:
    """构建默认 Tool Registry。"""

    registry = ToolRegistry()

    registry.register(
        ToolEntry(
            metadata=ToolMetadata(
                name="run_python",
                description="在持久 Python 命名空间中执行代码，用于复杂数据分析、转换和清洗。",
                capabilities=[
                    "python.execute",
                    "data.analyze",
                    "data.aggregate",
                    "data.clean",
                    "data.transform",
                ],
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
                capabilities=[
                    "workspace.list_data_files",
                    "file.list.data",
                    "file.inspect",
                ],
                risk_level="low",
                enabled=True,
            ),
            tool=list_data_files,
        )
    )

    registry.register(
        ToolEntry(
            metadata=ToolMetadata(
                name="list_workspace_files",
                description="列出当前工作区的项目目录结构，只读，不读取文件内容。",
                capabilities=[
                    "workspace.list_files",
                    "workspace.inspect_structure",
                    "file.list",
                    "file.inspect",
                ],
                risk_level="low",
                enabled=True,
            ),
            tool=list_workspace_files,
        )
    )

    registry.register(
        ToolEntry(
            metadata=ToolMetadata(
                name="resolve_workspace_file",
                description="根据文件名、路径或片段解析工作区内的候选文件路径，只读。",
                capabilities=[
                    "workspace.resolve_path",
                    "file.search",
                    "file.resolve",
                    "file.inspect",
                ],
                risk_level="low",
                enabled=True,
            ),
            tool=resolve_workspace_file,
        )
    )

    registry.register(
        ToolEntry(
            metadata=ToolMetadata(
                name="preview_data_file",
                description="安全预览工作区内数据文件的前几行，只读。",
                capabilities=[
                    "data.preview",
                    "file.read.preview",
                ],
                risk_level="low",
                enabled=True,
            ),
            tool=preview_data_file,
        )
    )

    return registry