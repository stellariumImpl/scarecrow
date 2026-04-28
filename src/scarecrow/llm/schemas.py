from typing import Literal

from pydantic import BaseModel, Field

ProviderName = Literal["openai", "deepseek", "ollama"]


class ChatModelSettings(BaseModel):
    """运行时构建 ChatModel 所需的标准化配置。

    这个类不是用户配置文件本身，而是 LLM Loader 的输入。
    后续可以继续扩展 base_url、timeout、max_tokens、fallback 等字段。
    """

    provider: ProviderName
    model: str
    api_key: str | None = None

    temperature: float = 0
    max_tokens: int | None = None
    timeout: float | None = None
    max_retries: int = 2

    base_url: str | None = None

    def requires_api_key(self) -> bool:
        return self.provider != "ollama"
