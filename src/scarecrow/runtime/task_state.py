# src/scarecrow/runtime/task_state.py

import re

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
        alias = _normalize_alias(alias)
        path = _normalize_source(path)
        description = _compact_text(description, max_chars=120)

        if not alias or not path:
            return

        existing = self.datasets.get(alias)

        if existing is not None:
            merged_description = description or existing.description
            self.datasets[alias] = DatasetState(
                alias=alias,
                path=path or existing.path,
                description=merged_description,
            )
            return

        self.datasets[alias] = DatasetState(
            alias=alias,
            path=path,
            description=description,
        )

    def add_finding(
        self,
        source: str,
        summary: str,
        confidence: float = 0.8,
    ) -> None:
        source = _normalize_source(source) or "observation"
        summary = _compact_text(summary, max_chars=180)

        if not summary:
            return

        if confidence < 0.6:
            return

        if len(summary) < 8:
            return

        new_key = (source.lower(), _normalize_text(summary))

        for finding in self.findings:
            existing_key = (
                finding.source.lower(),
                _normalize_text(finding.summary),
            )

            if existing_key == new_key:
                return

            if finding.source.lower() == source.lower() and _too_similar(
                finding.summary,
                summary,
            ):
                if confidence > finding.confidence and len(summary) > len(finding.summary):
                    finding.summary = summary
                    finding.confidence = confidence
                return

        self.findings.append(
            Finding(
                source=source,
                summary=summary,
                confidence=confidence,
            )
        )

        if len(self.findings) > 30:
            self.findings = self.findings[-30:]

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
            parts.append("\n最近发现结论：")
            for finding in self.findings[-12:]:
                parts.append(
                    f"- [{finding.source}] {finding.summary} "
                    f"(confidence={finding.confidence:.2f})"
                )

        if len(parts) == 1:
            return "## 当前任务状态\n\n暂无结构化任务状态。"

        return "\n".join(parts)


def _normalize_alias(alias: str) -> str:
    cleaned = alias.strip().strip("`'\"")
    cleaned = cleaned.replace("\\", "/")

    if "/" in cleaned:
        cleaned = cleaned.rsplit("/", 1)[-1]

    if "." in cleaned:
        cleaned = cleaned.rsplit(".", 1)[0]

    cleaned = re.sub(r"[^A-Za-z0-9_\-]+", "_", cleaned)
    cleaned = cleaned.strip("_-")

    return cleaned


def _normalize_source(source: str) -> str:
    cleaned = source.strip().strip("`'\"")
    cleaned = cleaned.replace("\\", "/")

    marker = "/data/"
    if marker in cleaned:
        return "data/" + cleaned.split(marker, 1)[1]

    marker = "/src/"
    if marker in cleaned:
        return "src/" + cleaned.split(marker, 1)[1]

    return cleaned


def _compact_text(text: str, max_chars: int) -> str:
    cleaned = " ".join(text.strip().split())

    if len(cleaned) <= max_chars:
        return cleaned

    return cleaned[: max_chars - 3] + "..."


def _normalize_text(text: str) -> str:
    cleaned = text.lower()
    cleaned = re.sub(r"\s+", "", cleaned)
    cleaned = re.sub(r"[，。；：,.;;:()\[\]【】`'\"]", "", cleaned)
    return cleaned


def _too_similar(a: str, b: str) -> bool:
    """轻量相似度判断，避免重复 finding 堆积。"""

    norm_a = _normalize_text(a)
    norm_b = _normalize_text(b)

    if not norm_a or not norm_b:
        return False

    if norm_a in norm_b or norm_b in norm_a:
        return True

    set_a = set(norm_a)
    set_b = set(norm_b)

    overlap = len(set_a & set_b)
    union = len(set_a | set_b)

    if union == 0:
        return False

    return overlap / union >= 0.88