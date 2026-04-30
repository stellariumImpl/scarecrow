# src/scarecrow/context/assembler.py

from pathlib import Path

from scarecrow.config import SKILLS_DIR
from scarecrow.context.schemas import ContextBuildInput
from scarecrow.skills import (
    build_skill_index,
    build_skill_prompt_block,
    load_skill_registry,
)


BASE_PROMPT = "你是 Scarecrow，一个本地数据分析助手。用中文回答用户。"


TOOL_USE_POLICY = """## 工具使用规则

- 只调用完成当前用户请求所必需的工具。
- 不要反复调用同一个工具查询相同参数。
- 禁止用同一个工具和同一组参数重复调用。
- 文件查找类请求中，只查找用户明确提到的文件名或路径片段。
- 如果文件解析工具返回没有找到，不要猜测其他文件名，不要转而查询无关文件。
- 如果工具已经返回“没有找到”，不要换成无关文件继续查找。
- 如果需要更多信息，直接向用户说明需要补充，而不是继续盲目搜索。
- 单轮回答中通常最多调用 2 次工具；除非用户明确要求深入扫描。

## 工具结果引用规则

- 回答必须严格基于本轮工具返回的结果。
- 如果只调用了 resolve_workspace_file 且结果是没有找到，只能说明“没有找到该文件”，不要推断“工作区没有任何数据文件”。
- 只有调用 list_data_files 且返回未发现数据文件时，才能说“当前工作区没有发现数据文件”。
- 只有调用 list_workspace_files 后，才能总结项目目录结构。
- 如果没有调用 preview_data_file，不要描述数据内容。
- 如果没有调用 run_python，不要声称已经完成统计、聚合、清洗或计算。
- 不要把 workspace brief 中的历史/静态信息当成本轮工具检查结果，除非明确说明“根据启动时扫描结果”。

## 当前任务状态使用规则

- 如果系统提供了“当前任务状态”，可以把它作为本轮回答的上下文依据。
- 当前任务状态只代表本次 REPL 会话中的短期分析上下文，不是永久记忆。
- 当用户说“刚才的数据”“基于前面结果”“给我一个方案”“不要写入文件”时，优先参考当前任务状态。
- 如果当前任务状态和本轮工具结果冲突，以本轮工具结果为准。
"""


class ContextAssembler:
    """负责组装 Agent system prompt。

    注意：
    - skills 模块只负责扫描、解析、注册 skill
    - context 模块负责决定哪些上下文进入 prompt
    """

    def __init__(self, skills_dir: Path = SKILLS_DIR) -> None:
        self.skills_dir = skills_dir

    def build_system_prompt(self, payload: ContextBuildInput) -> str:
        parts: list[str] = [BASE_PROMPT, TOOL_USE_POLICY]

        if payload.workspace is not None:
            parts.append(self._build_workspace_context(payload.workspace))

        if payload.task_state_brief:
            parts.append(payload.task_state_brief)

        registry = load_skill_registry(self.skills_dir)

        if payload.include_skill_index:
            parts.append(build_skill_index(registry))

        if payload.include_all_skills:
            skill_docs = registry.enabled_documents()
        else:
            selected = payload.selected_skills or []
            skill_docs = [
                doc
                for name in selected
                if (doc := registry.get(name)) is not None
            ]

        skill_block = build_skill_prompt_block(skill_docs)
        if skill_block:
            parts.append(skill_block)

        return "\n\n".join(parts)

    def _build_workspace_context(self, workspace: Path) -> str:
        from scarecrow.workspace import workspace_brief

        return workspace_brief(workspace)


def build_system_prompt(
    workspace: Path | None = None,
    selected_skills: list[str] | None = None,
    include_skill_index: bool = False,
    include_all_skills: bool = True,
    skills_dir: Path = SKILLS_DIR,
    task_state_brief: str | None = None,
) -> str:
    """便捷函数。

    当前用于兼容 repl.py / runtime。
    后续 runtime 可以直接使用 ContextAssembler。
    """

    assembler = ContextAssembler(skills_dir=skills_dir)
    return assembler.build_system_prompt(
        ContextBuildInput(
            workspace=workspace,
            selected_skills=selected_skills,
            include_skill_index=include_skill_index,
            include_all_skills=include_all_skills,
            task_state_brief=task_state_brief,
        )
    )