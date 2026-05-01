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

    # Plan Mode 是会话级开关，不属于任务状态。
    # /reset 不应关闭 Plan Mode。
    plan_mode: bool = False

    def reset(self) -> None:
        """清空当前任务上下文，但保留用户会话设置。"""

        self.messages = []
        self.agent = None
        self.task_state = TaskState()