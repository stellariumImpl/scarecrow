# src/scarecrow/runtime/task_state.py

from pydantic import BaseModel, Field


class DatasetState(BaseModel):
    """当前会话中已识别的数据集状态。"""

    alias: str
    path: str
    description: str = ""


class Finding(BaseModel):
    """当前会话中已发现的问题或结论。"""

    source: str
    summary: str
    confidence: float = 0.8


class TaskState(BaseModel):
    """当前 REPL session 的短期任务状态。

    注意：
    - 这是 session-level state，不是长期记忆
    - 由 ObservationExtractor 自动更新
    - reset 后清空
    """

    datasets: dict[str, DatasetState] = Field(default_factory=dict)
    findings: list[Finding] = Field(default_factory=list)

    def add_dataset(self, alias: str, path: str, description: str = "") -> None:
        alias = alias.strip()
        path = path.strip()

        if not alias or not path:
            return

        self.datasets[alias] = DatasetState(
            alias=alias,
            path=path,
            description=description.strip(),
        )

    def add_finding(
        self,
        source: str,
        summary: str,
        confidence: float = 0.8,
    ) -> None:
        source = source.strip() or "unknown"
        summary = summary.strip()

        if not summary:
            return

        key = (source.lower(), summary.lower())

        for finding in self.findings:
            existing_key = (
                finding.source.lower(),
                finding.summary.lower(),
            )
            if existing_key == key:
                return

        self.findings.append(
            Finding(
                source=source,
                summary=summary,
                confidence=confidence,
            )
        )

    def has_context(self) -> bool:
        return bool(self.datasets or self.findings)

    def brief(self) -> str:
        """生成可注入 prompt 的任务状态摘要。"""

        parts: list[str] = ["## 当前任务状态"]

        if self.datasets:
            parts.append("\n已识别数据集：")
            for dataset in self.datasets.values():
                desc = f" — {dataset.description}" if dataset.description else ""
                parts.append(f"- {dataset.alias}: `{dataset.path}`{desc}")

        if self.findings:
            parts.append("\n已发现结论：")
            for finding in self.findings[-12:]:
                parts.append(
                    f"- [{finding.source}] {finding.summary} "
                    f"(confidence={finding.confidence:.2f})"
                )

        if len(parts) == 1:
            return "## 当前任务状态\n\n暂无结构化任务状态。"

        return "\n".join(parts)