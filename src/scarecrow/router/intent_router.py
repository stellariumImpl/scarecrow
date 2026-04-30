# src/scarecrow/router/intent_router.py

"""Intent Router：将用户输入路由为结构化 RouteDecision。"""

import json
import re

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from scarecrow.router.schemas import RouteDecision


def _build_router_prompt(capability_index: str = "") -> str:
    """构造 Router system prompt。

    capability_index 由 ToolRegistry 动态生成，使 Router 根据能力而不是工具名决策。
    """

    return f"""你是 Scarecrow 的 Intent Router。

你的职责：
- 判断用户请求的主要意图 intent
- 判断完成请求需要哪些抽象能力 required_capabilities
- 输出严格 JSON
- 不执行任务
- 不回答用户问题
- 不编造不存在的能力
- 不输出 markdown
- 不输出 ```json 代码块
- 不输出长推理

你必须输出如下 JSON 结构：

{{
  "intent": "chat | data_analysis | file_inspection | code_debugging | config | unknown",
  "confidence": 0.0,
  "required_capabilities": [],
  "required_skills": [],
  "required_tools": [],
  "needs_clarification": false,
  "clarification_question": null,
  "risk_level": "low | medium | high",
  "reason": "一句简短原因"
}}

字段说明：

1. intent
用户请求的主意图。

2. confidence
你对路由判断的置信度，范围 0 到 1。

3. required_capabilities
完成任务所需的抽象能力，不是具体工具名，也不是具体 Skill 名。
例如：
- workspace.resolve_path
- workspace.list_data_files
- workspace.inspect_structure
- data.preview
- data.explore
- data.analyze
- data.clean
- python.execute

4. required_skills
过渡字段，默认输出 []。
Skill 是任务方法论，不等同于工具。
优先通过 required_capabilities 让系统自动选择 Skill。
只有当你非常确定某个 Skill 名称，并且该 Skill 是完成任务所需的方法论时，才填写 required_skills。
不要为了普通数据分析默认填写 Skill 名。

5. required_tools
过渡字段。默认输出 []。
不要主动填工具名。工具由系统根据 required_capabilities 自动选择。

6. needs_clarification
如果用户意图不明确、缺少文件名、缺少任务目标，设为 true。

7. clarification_question
需要澄清时，给用户的一句话问题。

8. risk_level
- low：只读、普通聊天、列文件、预览少量数据
- medium：执行 Python、清洗、转换、批量处理、可能修改内存状态
- high：删除文件、写文件、联网、执行 shell、访问密钥、危险操作

9. reason
一句简短说明为什么这样路由。不要写长推理。

意图说明：

1. chat
普通闲聊、解释概念、不需要读取文件、不需要执行 Python。

2. file_inspection
用户想列出文件、查看项目结构、查找某个文件、确认文件是否存在。

3. data_analysis
用户想预览、读取、统计、聚合、探索、清洗、转换数据文件。

4. code_debugging
用户提供代码或报错，想解释或修复。

5. config
用户想配置 provider、model、API key、LangSmith。

6. unknown
意图不清楚，或缺少必要信息。

能力选择参考：

- 用户问“当前有哪些数据文件 / 有没有 csv / 数据集在哪里”
  intent = "file_inspection"
  required_capabilities 通常包含 ["workspace.list_data_files"]
  required_skills = []
  required_tools = []
  risk_level = "low"

- 用户问“当前有什么文件 / 当前目录有什么 / 有哪些文件”
  intent = "file_inspection"
  required_capabilities 通常包含 ["workspace.list_files"]
  required_skills = []
  required_tools = []
  risk_level = "low"

- 用户问“项目结构 / 目录结构 / 工程结构 / 有哪些模块”
  intent = "file_inspection"
  required_capabilities 通常包含 ["workspace.inspect_structure"]
  required_skills = []
  required_tools = []
  risk_level = "low"

- 用户问“帮我找某个文件 / 某文件在哪里 / 有没有某个文件 / 某文件路径是什么”
  intent = "file_inspection"
  required_capabilities 通常包含 ["workspace.resolve_path"]
  required_skills = []
  required_tools = []
  risk_level = "low"

- 用户问“看一下某个数据文件前几行 / 预览数据 / 看看数据长什么样 / 查看列名”
  intent = "data_analysis"
  required_capabilities 通常包含 ["workspace.resolve_path", "data.preview"]
  required_skills = []
  required_tools = []
  risk_level = "low"

- 用户问“探索一下数据 / 做 EDA / 看 shape dtype 缺失值 分布”
  intent = "data_analysis"
  required_capabilities 通常包含 ["workspace.resolve_path", "python.execute", "data.explore", "data.profile"]
  required_skills 默认输出 []
  required_tools = []
  risk_level 通常为 "medium"

- 用户问“统计 / 聚合 / 相关性 / 缺失率 / 平均值 / 分组分析”
  intent = "data_analysis"
  required_capabilities 通常包含 ["workspace.resolve_path", "python.execute", "data.analyze"]
  required_skills 默认输出 []
  required_tools = []
  risk_level 通常为 "medium"

- 用户问“清洗 / 去重 / 处理缺失 / 转换字段 / 保存结果”
  intent = "data_analysis"
  required_capabilities 通常包含 ["workspace.resolve_path", "python.execute", "data.clean"]
  required_skills 默认输出 []
  required_tools = []
  risk_level 至少为 "medium"

- 用户问“换模型 / 设置 API key / 配置 DeepSeek / 配置 OpenAI / 开启 LangSmith”
  intent = "config"
  required_capabilities = []
  required_skills = []
  required_tools = []
  risk_level = "low"

- 用户提供 Python 报错、traceback、代码片段并要求修复
  intent = "code_debugging"
  required_capabilities 通常为空，除非明确需要执行代码
  required_skills = []
  required_tools = []
  risk_level 通常为 "medium"

工具和 Skill 选择原则：
- required_capabilities 是主输出。
- Tool 由系统根据 required_capabilities 自动选择。
- Skill 也会逐步由系统根据 required_capabilities 自动选择。
- required_skills 是过渡字段，不要默认填写。
- required_tools 是过渡字段，不要默认填写。
- 简单数据预览优先使用 data.preview，不要默认要求 python.execute。
- 复杂分析、清洗、聚合才需要 python.execute。
- 文件查找类请求不要猜测其他文件名。
- 用户让找某个文件名或路径片段时，只查找该片段，不要转而查询其他无关文件。
- 如果工具返回没有找到，应直接告诉用户没有找到，并建议用户确认文件名或查看项目结构。
- 如果用户只是问项目结构或列文件，不要要求 python.execute。
- 如果用户只是普通聊天，不要要求任何 capability。
- 如果不确定用户要找哪个文件，应澄清，而不是猜测。

{capability_index}
"""


class IntentRouter:
    """基于 JSON 输出 + Pydantic 校验的意图路由器。

    不使用 with_structured_output，避免 deepseek-reasoner / 部分 Ollama 模型
    不支持 tool_choice / structured output 时直接报错。
    """

    def __init__(self, model: BaseChatModel, capability_index: str = "") -> None:
        self.model = model
        self.capability_index = capability_index

    def route(self, user_input: str) -> RouteDecision:
        """返回结构化路由决策。"""

        system_prompt = _build_router_prompt(self.capability_index)

        msg = self.model.invoke(
            [
                SystemMessage(content=system_prompt),
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
            r"^```(?:json)?",
            "",
            cleaned.strip(),
            flags=re.IGNORECASE,
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

    注意：
    - 这是防止坏 JSON 导致 REPL 崩溃的兜底。
    - 这里允许少量稳定启发式，但不要把它当主路由逻辑。
    - 主路由逻辑仍应来自 LLM + capability index。
    """

    lowered = user_input.lower()

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

    data_preview_keywords = [
        "前 5 行",
        "前5行",
        "前几行",
        "head",
        "preview",
        "预览",
        "列名",
        "长什么样",
        "看看数据",
    ]

    data_explore_keywords = [
        "探索",
        "eda",
        "profile",
        "概览",
        "整体情况",
        "数据情况",
        "了解一下数据",
    ]

    data_clean_keywords = [
        "清洗",
        "去重",
        "处理缺失",
        "缺失处理",
        "转换字段",
        "字段类型",
        "保存结果",
    ]

    data_analysis_keywords = [
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
        "相关性",
        "分组",
        "聚合",
        "平均值",
        "均值",
    ]

    structure_keywords = [
        "项目结构",
        "目录结构",
        "工程结构",
        "有哪些模块",
        "结构",
        "tree",
    ]

    file_list_keywords = [
        "当前目录",
        "有哪些文件",
        "文件列表",
        "当前有什么文件",
        "目录里有什么",
    ]

    data_file_list_keywords = [
        "数据文件",
        "csv 文件",
        "csv文件",
        "有哪些 csv",
        "有哪些csv",
        "数据集",
    ]

    file_search_keywords = [
        "找",
        "查找",
        "在哪里",
        "有没有",
        "路径",
    ]

    if any(k in lowered for k in config_keywords):
        return RouteDecision(
            intent="config",
            confidence=0.55,
            required_capabilities=[],
            required_skills=[],
            required_tools=[],
            risk_level="low",
            reason="Router 输出解析失败，按配置相关关键词保守兜底。",
        )

    if any(k in lowered for k in debug_keywords):
        return RouteDecision(
            intent="code_debugging",
            confidence=0.55,
            required_capabilities=[],
            required_skills=[],
            required_tools=[],
            risk_level="medium",
            reason="Router 输出解析失败，按调试相关关键词保守兜底。",
        )

    if any(k in lowered for k in data_preview_keywords):
        return RouteDecision(
            intent="data_analysis",
            confidence=0.55,
            required_capabilities=[
                "workspace.resolve_path",
                "data.preview",
            ],
            required_skills=[],
            required_tools=[],
            risk_level="low",
            reason="Router 输出解析失败，按数据预览相关关键词保守兜底。",
        )

    if any(k in lowered for k in data_explore_keywords):
        return RouteDecision(
            intent="data_analysis",
            confidence=0.55,
            required_capabilities=[
                "workspace.resolve_path",
                "python.execute",
                "data.explore",
                "data.profile",
            ],
            required_skills=[],
            required_tools=[],
            risk_level="medium",
            reason="Router 输出解析失败，按数据探索相关关键词保守兜底。",
        )

    if any(k in lowered for k in data_clean_keywords):
        return RouteDecision(
            intent="data_analysis",
            confidence=0.55,
            required_capabilities=[
                "workspace.resolve_path",
                "python.execute",
                "data.clean",
            ],
            required_skills=[],
            required_tools=[],
            risk_level="medium",
            reason="Router 输出解析失败，按数据清洗相关关键词保守兜底。",
        )

    if any(k in lowered for k in data_analysis_keywords):
        return RouteDecision(
            intent="data_analysis",
            confidence=0.55,
            required_capabilities=[
                "workspace.resolve_path",
                "python.execute",
                "data.analyze",
            ],
            required_skills=[],
            required_tools=[],
            risk_level="medium",
            reason="Router 输出解析失败，按数据分析相关关键词保守兜底。",
        )

    if any(k in lowered for k in structure_keywords):
        return RouteDecision(
            intent="file_inspection",
            confidence=0.55,
            required_capabilities=["workspace.inspect_structure"],
            required_skills=[],
            required_tools=[],
            risk_level="low",
            reason="Router 输出解析失败，按项目结构查看相关关键词保守兜底。",
        )

    if any(k in lowered for k in data_file_list_keywords):
        return RouteDecision(
            intent="file_inspection",
            confidence=0.55,
            required_capabilities=["workspace.list_data_files"],
            required_skills=[],
            required_tools=[],
            risk_level="low",
            reason="Router 输出解析失败，按数据文件列表相关关键词保守兜底。",
        )

    if any(k in lowered for k in file_list_keywords):
        return RouteDecision(
            intent="file_inspection",
            confidence=0.55,
            required_capabilities=["workspace.list_files"],
            required_skills=[],
            required_tools=[],
            risk_level="low",
            reason="Router 输出解析失败，按文件列表相关关键词保守兜底。",
        )

    if any(k in lowered for k in file_search_keywords):
        return RouteDecision(
            intent="file_inspection",
            confidence=0.55,
            required_capabilities=["workspace.resolve_path"],
            required_skills=[],
            required_tools=[],
            risk_level="low",
            reason="Router 输出解析失败，按文件查找相关关键词保守兜底。",
        )

    return RouteDecision(
        intent="chat",
        confidence=0.5,
        required_capabilities=[],
        required_skills=[],
        required_tools=[],
        risk_level="low",
        reason="Router 输出解析失败，默认按普通聊天处理。",
    )