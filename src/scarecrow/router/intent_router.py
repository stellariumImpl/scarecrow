import json
import re

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from scarecrow.router.schemas import RouteDecision

_ROUTER_SYSTEM_PROMPT = """你是 Scarecrow 的 Intent Router。

你的职责：
- 只判断用户请求的意图
- 只输出 JSON
- 不执行任务
- 不回答用户问题
- 不编造不存在的工具
- 不输出 markdown
- 不输出 ```json 代码块
- 不输出长推理

你必须输出如下 JSON 结构：

{
  "intent": "chat | data_analysis | file_inspection | code_debugging | config | unknown",
  "confidence": 0.0,
  "required_skills": [],
  "required_tools": [],
  "needs_clarification": false,
  "clarification_question": null,
  "risk_level": "low | medium | high",
  "reason": "一句简短原因"
}

可用 intent：

1. chat
普通闲聊、解释概念、不需要读取文件、不需要执行 Python 的问题。

2. data_analysis
用户想读取、统计、清洗、探索、分析数据文件或 DataFrame。
典型表达：
- 看一下 xxx.csv
- 探索这份数据
- 统计某列
- 缺失值怎么样
- 相关性怎么样
- 分组聚合
- 清洗数据

3. file_inspection
用户想查看当前工作区有哪些文件、确认文件是否存在、列出数据文件。
典型表达：
- 当前目录有什么
- 帮我找数据文件
- 看看有哪些 csv
- 文件路径是什么

4. code_debugging
用户提供报错、代码片段，想让你解释或修复。
典型表达：
- 这个报错是什么意思
- 为什么运行失败
- 帮我改这段代码

5. config
用户想配置 provider、model、api key、LangSmith、环境设置。
典型表达：
- 配置模型
- 换成 deepseek
- 设置 API key
- 开启 LangSmith

6. unknown
意图不清楚，或缺少必要信息。

工具与 skill 建议规则：

- data_analysis:
  required_tools 通常包含 ["run_python"]
  required_skills 通常包含 ["run-python"]
  如果是系统化探索数据，还应包含 ["data-explorer"]

- file_inspection:
  required_tools 通常包含 ["list_data_files"]
  required_skills 通常为空
  只需要列出工作区数据文件时，不要使用 run_python
  如果用户明确要求读取文件内容或统计数据，则转为 data_analysis

- chat:
  通常不需要 tool 和 skill。

- config:
  通常不需要 tool 和 skill，因为 CLI 命令处理。

风险等级：
- low：普通聊天、只读分析、查看数据概览
- medium：数据清洗、批量修改、生成代码、可能覆盖变量
- high：删除文件、写入文件、联网、执行 shell、泄露密钥、危险代码

澄清规则：
- 如果用户只说“分析一下”，但没有任何可用上下文，可以 needs_clarification=true
- 如果用户提到明确文件名或当前工作区已有明显数据文件，可以不澄清
- 如果用户请求危险操作，应提高 risk_level，必要时澄清

reason 只写一句简短说明。
"""


class IntentRouter:
    """基于 JSON 输出 + Pydantic 校验的意图路由器。

    不使用 with_structured_output，避免 deepseek-reasoner / 部分 Ollama 模型
    不支持 tool_choice / structured output 时直接报错。
    """

    def __init__(self, model: BaseChatModel) -> None:
        self.model = model

    def route(self, user_input: str) -> RouteDecision:
        """返回结构化路由决策。"""

        msg = self.model.invoke(
            [
                SystemMessage(content=_ROUTER_SYSTEM_PROMPT),
                HumanMessage(content=user_input),
            ]
        )

        content = _message_content_to_text(msg.content)
        data = _extract_json_object(content)

        try:
            return RouteDecision.model_validate(data)
        except Exception:
            return _fallback_route(user_input, raw_output=content)


def _message_content_to_text(content) -> str:
    """把 LangChain message.content 统一转成字符串。"""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text:
                    parts.append(str(text))
            else:
                parts.append(str(item))
        return "\n".join(parts)

    return str(content)


def _extract_json_object(text: str) -> dict:
    """从模型输出中提取 JSON object。

    兼容这些情况：
    - 纯 JSON
    - ```json ... ```
    - 前后夹杂少量说明文字
    """

    cleaned = text.strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(
            r"^```(?:json)?", "", cleaned.strip(), flags=re.IGNORECASE
        ).strip()
        cleaned = re.sub(r"```$", "", cleaned.strip()).strip()

    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in router output: {text}")

    data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("Router output JSON is not an object.")

    return data


def _fallback_route(user_input: str, raw_output: str = "") -> RouteDecision:
    """Router 解析失败时的保守兜底。

    注意：这不是主路由逻辑，只是防止模型输出坏 JSON 导致 REPL 崩。
    """

    lowered = user_input.lower()

    data_keywords = [
        ".csv",
        ".parquet",
        ".jsonl",
        ".xlsx",
        "dataframe",
        "df",
        "数据",
        "缺失值",
        "统计",
        "分析",
        "探索",
        "清洗",
        "相关性",
        "分组",
        "聚合",
    ]

    config_keywords = [
        "api key",
        "apikey",
        "provider",
        "model",
        "模型",
        "配置",
        "deepseek",
        "openai",
        "ollama",
        "langsmith",
    ]

    debug_keywords = [
        "error",
        "exception",
        "traceback",
        "报错",
        "错误",
        "运行失败",
        "bug",
        "debug",
    ]

    file_keywords = [
        "当前目录",
        "有哪些文件",
        "文件列表",
        "找文件",
        "数据文件",
        "csv 文件",
    ]

    if any(k in lowered for k in config_keywords):
        return RouteDecision(
            intent="config",
            confidence=0.55,
            risk_level="low",
            reason="Router 输出解析失败，按配置相关关键词保守兜底。",
        )

    if any(k in lowered for k in debug_keywords):
        return RouteDecision(
            intent="code_debugging",
            confidence=0.55,
            risk_level="medium",
            reason="Router 输出解析失败，按调试相关关键词保守兜底。",
        )

    if any(k in lowered for k in file_keywords):
        return RouteDecision(
            intent="file_inspection",
            confidence=0.55,
            risk_level="low",
            reason="Router 输出解析失败，按文件查看相关关键词保守兜底。",
        )

    if any(k in lowered for k in data_keywords):
        return RouteDecision(
            intent="data_analysis",
            confidence=0.55,
            required_skills=["run-python"],
            required_tools=["run_python"],
            risk_level="low",
            reason="Router 输出解析失败，按数据分析相关关键词保守兜底。",
        )

    return RouteDecision(
        intent="chat",
        confidence=0.5,
        risk_level="low",
        reason="Router 输出解析失败，默认按普通聊天处理。",
    )
