from pathlib import Path

from scarecrow.config import SKILLS_DIR
from scarecrow.context.schemas import ContextBuildInput
from scarecrow.skills import (
    build_skill_index,
    build_skill_prompt_block,
    load_skill_registry,
)


BASE_PROMPT = "你是 Scarecrow，一个本地数据分析助手。用中文回答用户。"


class ContextAssembler:
    """负责组装 Agent system prompt。

    注意：
    - skills 模块只负责扫描、解析、注册 skill
    - context 模块负责决定哪些上下文进入 prompt
    """

    def __init__(self, skills_dir: Path = SKILLS_DIR) -> None:
        self.skills_dir = skills_dir

    def build_system_prompt(self, payload: ContextBuildInput) -> str:
        parts: list[str] = [BASE_PROMPT]

        if payload.workspace is not None:
            parts.append(self._build_workspace_context(payload.workspace))

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
) -> str:
    """便捷函数。

    当前用于兼容 repl.py。
    后续 runtime 可以直接使用 ContextAssembler。
    """

    assembler = ContextAssembler(skills_dir=skills_dir)
    return assembler.build_system_prompt(
        ContextBuildInput(
            workspace=workspace,
            selected_skills=selected_skills,
            include_skill_index=include_skill_index,
            include_all_skills=include_all_skills,
        )
    )