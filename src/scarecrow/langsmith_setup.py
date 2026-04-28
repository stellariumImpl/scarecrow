# src/scarecrow/langsmith_setup.py

"""LangSmith 追踪：根据配置设置环境变量"""

import os

from scarecrow.config import LangSmithConfig, load_langsmith_config


def apply_langsmith_env() -> tuple[bool, str]:
    """
    根据用户配置设置 LangSmith 环境变量。
    LangChain 1.0 检测到这些环境变量后会自动开启追踪，无需修改 agent 代码。

    返回: (是否启用, 项目名)
    """
    cfg = load_langsmith_config()
    if cfg is None or not cfg.enabled or not cfg.api_key:
        # 显式关掉，避免之前 export 的环境变量残留
        os.environ.pop("LANGSMITH_TRACING", None)
        os.environ.pop("LANGCHAIN_TRACING_V2", None)
        return False, ""

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = cfg.api_key
    os.environ["LANGSMITH_PROJECT"] = cfg.project
    # 旧名兼容：部分包仍然读这些变量
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = cfg.api_key
    os.environ["LANGCHAIN_PROJECT"] = cfg.project
    return True, cfg.project
