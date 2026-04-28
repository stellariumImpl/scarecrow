from scarecrow.llm.schemas import ProviderName

PROVIDER_MODEL_PREFIX: dict[ProviderName, str] = {
    "openai": "openai",
    "deepseek": "deepseek",
    "ollama": "ollama",
}


PROVIDER_REQUIRES_API_KEY: dict[ProviderName, bool] = {
    "openai": True,
    "deepseek": True,
    "ollama": False,
}


def build_model_id(provider: ProviderName, model: str) -> str:
    """构造 LangChain init_chat_model 可识别的模型 ID。

    例如：
    - openai:gpt-4o-mini
    - deepseek:deepseek-chat
    - ollama:llama3.1
    """
    prefix = PROVIDER_MODEL_PREFIX[provider]
    return f"{prefix}:{model}"


def provider_requires_api_key(provider: ProviderName) -> bool:
    return PROVIDER_REQUIRES_API_KEY[provider]
