from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

SkillSource = Literal["builtin", "user", "project"]


class SkillMetadata(BaseModel):
    """Skill 的轻量元数据。

    这部分信息适合进入 Router / Skill Index。
    不应该把完整 SKILL.md 正文都塞进 Router。
    """

    name: str = Field(..., min_length=1)
    description: str = ""
    path: Path
    source: SkillSource = "user"
    enabled: bool = True


class SkillDocument(BaseModel):
    """完整 Skill 文档。"""

    metadata: SkillMetadata
    content: str
