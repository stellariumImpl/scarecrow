# src/scarecrow/skills/registry.py

from pathlib import Path

from scarecrow.skills.parser import parse_skill_file
from scarecrow.skills.schemas import SkillDocument, SkillMetadata, SkillSource


class SkillRegistry:
    """Skill 注册表。

    负责扫描、解析、去重、查询 Skill。

    capabilities 是可选增强：
    - 声明 capabilities 的 skill 可以被自动匹配
    - 未声明 capabilities 的 skill 仍然可通过 required_skills 被选中
    """

    def __init__(self) -> None:
        self._skills: dict[str, SkillDocument] = {}

    def register(self, skill: SkillDocument) -> None:
        """注册一个 skill。

        同名 skill 后注册的会覆盖先注册的。
        未来可以实现优先级：
        user > project > builtin
        """

        self._skills[skill.metadata.name] = skill

    def scan_dir(self, skills_dir: Path, source: SkillSource = "user") -> None:
        """递归扫描目录下所有 SKILL.md。"""

        if not skills_dir.exists():
            return

        for skill_path in sorted(skills_dir.rglob("SKILL.md")):
            skill = parse_skill_file(skill_path, source=source)

            if skill is None:
                continue

            self.register(skill)

    def list_metadata(self) -> list[SkillMetadata]:
        return [skill.metadata for skill in self._skills.values()]

    def list_documents(self) -> list[SkillDocument]:
        return list(self._skills.values())

    def get(self, name: str) -> SkillDocument | None:
        return self._skills.get(name)

    def enabled_documents(self) -> list[SkillDocument]:
        return [
            skill
            for skill in self._skills.values()
            if skill.metadata.enabled
        ]

    def supported_capabilities(self) -> set[str]:
        """返回所有 enabled skill 声明的 capability。"""

        capabilities: set[str] = set()

        for skill in self._skills.values():
            if not skill.metadata.enabled:
                continue

            capabilities.update(skill.metadata.capabilities)

        return capabilities

    def select_skill_names_by_capabilities(
        self,
        required_capabilities: list[str],
    ) -> list[str]:
        """根据 required_capabilities 选择 skill 名称。

        渐进式原则：
        - 没有声明 capabilities 的 skill 不会被自动选中
        - 但它们仍然可以通过 Router.required_skills 被选中
        """

        if not required_capabilities:
            return []

        required = set(required_capabilities)
        selected: list[str] = []

        for skill in self._skills.values():
            meta = skill.metadata

            if not meta.enabled:
                continue

            if required.intersection(meta.capabilities):
                selected.append(meta.name)

        return selected