# src/scarecrow/config.py

"""配置管理：读写 ~/.scarecrow/config.json"""

import json
from pathlib import Path
from typing import Optional

PROVIDER_MODELS: dict[str, list[str]] = {
    "openai": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
    "deepseek": ["deepseek-chat", "deepseek-reasoner"],
    "ollama": [],
}

PROVIDER_LABELS: dict[str, str] = {
    "openai": "OpenAI",
    "deepseek": "DeepSeek",
    "ollama": "Ollama (local)",
}

CONFIG_DIR = Path.home() / ".scarecrow"
CONFIG_FILE = CONFIG_DIR / "config.json"
SKILLS_DIR = CONFIG_DIR / "skills"


class LLMConfig:
    def __init__(self, provider: str, model: str, api_key: str):
        self.provider = provider
        self.model = model
        self.api_key = api_key

    def masked_key(self) -> str:
        if len(self.api_key) <= 8:
            return "*" * len(self.api_key)
        return f"{self.api_key[:4]}...{self.api_key[-4:]}"

    def to_dict(self) -> dict:
        return {
            "provider": self.provider,
            "model": self.model,
            "api_key": self.api_key,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LLMConfig":
        provider = d.get("provider", "openai")
        if provider not in PROVIDER_MODELS:
            provider = "openai"
        return cls(
            provider=provider,
            model=d.get("model", "gpt-4o"),
            api_key=d.get("api_key", ""),
        )


class LangSmithConfig:
    """LangSmith 追踪配置，全部可选"""

    def __init__(self, api_key: str, project: str = "scarecrow", enabled: bool = True):
        self.api_key = api_key
        self.project = project
        self.enabled = enabled

    def masked_key(self) -> str:
        if len(self.api_key) <= 8:
            return "*" * len(self.api_key)
        return f"{self.api_key[:4]}...{self.api_key[-4:]}"

    def to_dict(self) -> dict:
        return {
            "api_key": self.api_key,
            "project": self.project,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LangSmithConfig":
        return cls(
            api_key=d.get("api_key", ""),
            project=d.get("project", "scarecrow"),
            enabled=d.get("enabled", True),
        )


# ---------------------------------------------------------------------------
# 整体配置文件结构（向后兼容旧版本扁平结构）
# {
#   "llm": {...},
#   "langsmith": {...}
# }
# ---------------------------------------------------------------------------

def _read_config_file() -> dict:
    """读取整个配置文件，兼容旧的扁平 LLM 结构"""
    if not CONFIG_FILE.exists():
        return {}
    try:
        data = json.loads(CONFIG_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return {}

    if not isinstance(data, dict):
        return {}

    # 兼容旧版本：顶层就是 LLM 字段（provider/model/api_key）
    if "provider" in data and "llm" not in data:
        return {"llm": data}
    return data


def _write_config_file(data: dict) -> None:
    ensure_config_dir()
    CONFIG_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    CONFIG_FILE.chmod(0o600)


def load_config() -> Optional[LLMConfig]:
    data = _read_config_file()
    llm = data.get("llm")
    if llm:
        return LLMConfig.from_dict(llm)
    return None


def save_config(config: LLMConfig) -> None:
    data = _read_config_file()
    data["llm"] = config.to_dict()
    _write_config_file(data)


def load_langsmith_config() -> Optional[LangSmithConfig]:
    data = _read_config_file()
    ls = data.get("langsmith")
    if ls:
        return LangSmithConfig.from_dict(ls)
    return None


def save_langsmith_config(config: LangSmithConfig) -> None:
    data = _read_config_file()
    data["langsmith"] = config.to_dict()
    _write_config_file(data)


def clear_langsmith_config() -> None:
    data = _read_config_file()
    data.pop("langsmith", None)
    _write_config_file(data)


def ensure_config_dir() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)