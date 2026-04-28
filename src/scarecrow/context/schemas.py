from pathlib import Path

from pydantic import BaseModel, Field


class ContextBuildInput(BaseModel):
    """构建 system prompt 所需的输入。"""

    workspace: Path | None = None

    selected_skills: list[str] | None = None

    include_skill_index: bool = False

    include_all_skills: bool = True