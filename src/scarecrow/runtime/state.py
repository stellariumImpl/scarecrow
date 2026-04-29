# src/scarecrow/runtime/state.py

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SessionState(BaseModel):
    """REPL 会话状态。

    注意：
    - messages 是 LangGraph / LangChain agent 的对话状态
    - agent 当前仍保留为 Any，因为 create_agent 返回对象类型较复杂
    - workspace 是当前 CLI 启动目录
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    workspace: Path
    messages: list = Field(default_factory=list)
    agent: Any | None = None

    def reset(self) -> None:
        self.messages = []
        self.agent = None