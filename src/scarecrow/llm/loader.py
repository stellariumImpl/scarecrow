from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel

from scarecrow.config import LLMConfig
from scarecrow.llm.registry import build_model_id, provider_requires_api_key
from scarecrow.llm.schemas import ChatModelSettings


def settings_from_config(config: LLMConfig) -> ChatModelSettings:
    """把用户配置转换为 LLM Loader 的标准输入。"""

    return ChatModelSettings(
        provider=config.provider,  # type: ignore[arg-type]
        model=config.model,
        api_key=config.api_key or None,
        temperature=0,
    )


def load_chat_model(settings: ChatModelSettings) -> BaseChatModel:
    """根据标准化配置创建 LangChain ChatModel。

    这个函数是项目里唯一应该直接调用 init_chat_model 的地方。
    """

    if provider_requires_api_key(settings.provider) and not settings.api_key:
        raise ValueError(f"{settings.provider} requires an API key.")

    model_id = build_model_id(settings.provider, settings.model)

    model_kwargs: dict = {
        "temperature": settings.temperature,
    }

    if settings.max_tokens is not None:
        model_kwargs["max_tokens"] = settings.max_tokens

    if settings.timeout is not None:
        model_kwargs["timeout"] = settings.timeout

    if settings.max_retries is not None:
        model_kwargs["max_retries"] = settings.max_retries

    if settings.base_url is not None:
        model_kwargs["base_url"] = settings.base_url

    if provider_requires_api_key(settings.provider):
        model_kwargs["api_key"] = settings.api_key

    return init_chat_model(model_id, **model_kwargs)


def load_chat_model_from_config(config: LLMConfig) -> BaseChatModel:
    """从用户配置直接创建 ChatModel。"""

    settings = settings_from_config(config)
    return load_chat_model(settings)
