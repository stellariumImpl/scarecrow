# src/scarecrow/tools/schemas.py

from typing import Literal

from pydantic import BaseModel, Field


ToolRiskLevel = Literal["low", "medium", "high"]


class ToolMetadata(BaseModel):
    """Tool 的轻量元数据。

    capabilities 是工具选择的核心：
    Router 输出 required_capabilities，
    ToolRegistry 根据 capabilities 选择工具。
    """

    name: str = Field(..., min_length=1)
    description: str = ""
    capabilities: list[str] = Field(default_factory=list)
    risk_level: ToolRiskLevel = "low"
    enabled: bool = True