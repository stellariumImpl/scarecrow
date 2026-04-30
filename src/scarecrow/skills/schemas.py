# src/scarecrow/skills/schemas.py

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


SkillSource = Literal["builtin", "user", "project"]


class SkillMetadata(BaseModel):
    """Skill 的轻量元数据。

    capabilities 是可选增强字段：
    - 旧版 skill 可以没有 capabilities
    - 新版 skill 可以声明 capabilities，用于自动匹配
    """

    name: str = Field(..., min_length=1)
    description: str = ""
    capabilities: list[str] = Field(default_factory=list)
    path: Path
    source: SkillSource = "user"
    enabled: bool = True


class SkillDocument(BaseModel):
    """完整 Skill 文档。"""

    metadata: SkillMetadata
    content: str