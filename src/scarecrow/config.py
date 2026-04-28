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
        # 如果 provider 不合法，fallback 到 openai
        if provider not in PROVIDER_MODELS:
            provider = "openai"
        return cls(
            provider=provider,
            model=d.get("model", "gpt-4o"),
            api_key=d.get("api_key", ""),
        )


def load_config() -> Optional[LLMConfig]:
    if CONFIG_FILE.exists():
        try:
            data = json.loads(CONFIG_FILE.read_text())
            if data:
                return LLMConfig.from_dict(data)
        except (json.JSONDecodeError, Exception):
            pass
    return None


def save_config(config: LLMConfig) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(config.to_dict(), indent=2, ensure_ascii=False))
    CONFIG_FILE.chmod(0o600)
