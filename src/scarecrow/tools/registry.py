# src/scarecrow/tools/registry.py

"""Tool Registry：集中注册工具，并基于 capability 选择工具。"""

from collections.abc import Callable
from dataclasses import dataclass

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
    """注册表中的单个工具条目。"""

    metadata: ToolMetadata
    tool: BaseTool | Callable


class ToolRegistry:
    """Tool 注册表。

    设计原则：
    - 工具通过 metadata.capabilities 声明自己能做什么
    - Router 输出 required_capabilities
    - Runtime 调用 registry 按 capability 自动选择工具
    - Runtime 不写死 intent/task_type 到具体工具的映射
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolEntry] = {}

    def register(self, entry: ToolEntry) -> None:
        """注册一个工具。"""

        self._tools[entry.metadata.name] = entry

    def get(self, name: str) -> ToolEntry | None:
        """按工具名获取工具条目。"""

        return self._tools.get(name)

    def list_metadata(self) -> list[ToolMetadata]:
        """列出所有工具元数据。"""

        return [entry.metadata for entry in self._tools.values()]

    def list_capabilities(self) -> list[str]:
        """列出所有已启用工具支持的 capability。"""

        return sorted(self.supported_capabilities())

    def supported_capabilities(self) -> set[str]:
        """返回当前启用工具支持的 capability 集合。"""

        capabilities: set[str] = set()

        for entry in self._tools.values():
            if not entry.metadata.enabled:
                continue

            capabilities.update(entry.metadata.capabilities)

        return capabilities

    def validate_capabilities(
        self,
        capabilities: list[str],
    ) -> tuple[list[str], list[str]]:
        """校验 Router 输出的 capabilities。

        返回：
        - known: 当前工具系统支持的能力
        - unknown: 当前工具系统不支持的能力

        同时会保持输入顺序并去重。
        """

        supported = self.supported_capabilities()

        known: list[str] = []
        unknown: list[str] = []

        for capability in capabilities:
            if capability in supported:
                known.append(capability)
            else:
                unknown.append(capability)

        return list(dict.fromkeys(known)), list(dict.fromkeys(unknown))

    def build_capability_index(self) -> str:
        """生成给 Router 使用的能力索引文本。"""

        lines = ["## 可用工具能力"]

        for entry in self._tools.values():
            meta = entry.metadata

            if not meta.enabled:
                continue

            capabilities = ", ".join(meta.capabilities) if meta.capabilities else "(无)"

            lines.append("")
            lines.append(f"- 工具: `{meta.name}`")
            lines.append(f"  描述: {meta.description}")
            lines.append(f"  风险: {meta.risk_level}")
            lines.append(f"  能力: {capabilities}")

        return "\n".join(lines)

    def select_tools(self, names: list[str]) -> list[BaseTool | Callable]:
        """根据工具名选择启用工具。

        这个方法保留给调试和过渡兼容逻辑使用。
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
        - 未知 capability 会被忽略
        """

        if not required_capabilities:
            return []

        known_capabilities, _ = self.validate_capabilities(required_capabilities)

        if not known_capabilities:
            return []

        required = set(known_capabilities)
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
    """判断工具风险是否在允许范围内。"""

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