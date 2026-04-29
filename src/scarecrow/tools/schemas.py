from typing import Literal

from pydantic import BaseModel, Field


ToolRiskLevel = Literal["low", "medium", "high"]


class ToolMetadata(BaseModel):
    """Tool 的轻量元数据。"""

    name: str = Field(..., min_length=1)
    description: str = ""
    risk_level: ToolRiskLevel = "low"
    enabled: bool = True