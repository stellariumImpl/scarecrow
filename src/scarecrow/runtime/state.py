# src/scarecrow/runtime/state.py

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from scarecrow.runtime.task_state import TaskState


class SessionState(BaseModel):
    """REPL 会话状态。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    workspace: Path
    messages: list = Field(default_factory=list)
    agent: Any | None = None
    task_state: TaskState = Field(default_factory=TaskState)

    def reset(self) -> None:
        self.messages = []
        self.agent = None
        self.task_state = TaskState()