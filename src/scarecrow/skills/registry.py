from pathlib import Path

from scarecrow.skills.parser import parse_skill_file
from scarecrow.skills.schemas import SkillDocument, SkillMetadata, SkillSource


class SkillRegistry:
    """Skill 注册表。

    负责扫描、解析、去重、查询 Skill。
    后续 Skill Router 会基于这个 registry 选择候选 skill。
    """

    def __init__(self) -> None:
        self._skills: dict[str, SkillDocument] = {}

    def register(self, skill: SkillDocument) -> None:
        """注册一个 skill。

        同名 skill 后注册的会覆盖先注册的。
        这样后面可以实现优先级：
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
        return [skill for skill in self._skills.values() if skill.metadata.enabled]
