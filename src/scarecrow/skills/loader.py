from pathlib import Path

from scarecrow.skills.registry import SkillRegistry
from scarecrow.skills.schemas import SkillDocument


def load_skill_registry(skills_dir: Path) -> SkillRegistry:
    """加载用户级 Skill Registry。"""

    registry = SkillRegistry()
    registry.scan_dir(skills_dir, source="user")
    return registry


def build_skill_index(registry: SkillRegistry) -> str:
    """构建轻量 Skill Index。

    后续 Router 应该优先看到这个，而不是完整 SKILL.md。
    """

    metadata = registry.list_metadata()

    if not metadata:
        return "## 可用 Skills\n\n当前没有可用 Skill。"

    lines = ["## 可用 Skills"]

    for item in metadata:
        lines.append(f"- `{item.name}`: {item.description}")

    return "\n".join(lines)


def build_skill_prompt_block(skills: list[SkillDocument]) -> str:
    """把完整 Skill 文档拼成 prompt block。"""

    if not skills:
        return ""

    blocks = [
        skill.content
        for skill in skills
        if skill.metadata.enabled and skill.content
    ]

    if not blocks:
        return ""

    return "--- 可用能力 ---\n\n" + "\n\n".join(blocks)