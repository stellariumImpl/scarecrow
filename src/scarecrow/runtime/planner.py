# src/scarecrow/runtime/planner.py

"""Task Planner.

Planner 负责把用户请求解释成可执行计划。

设计原则：
- Planner 负责语义判断和计划设计
- Runtime 负责执行计划
- Router 负责单个 step 的 capability 判断
- Planner 不选择工具
- Planner 不输出 capability
- Planner 不写代码
- Planner 不执行任务
- Planner 只输出结构化 TaskPlan
"""

import json
import re
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, field_validator

from scarecrow.config import LLMConfig
from scarecrow.llm import load_chat_model_from_config


PlanMode = Literal["single_task", "explicit_multi_task", "planned_task"]

InputInterpretation = Literal[
    "single_operation_ready",
    "single_operation_missing_input",
    "explicit_multi_task",
    "open_ended_goal",
    "collection_goal",
]

TargetScope = Literal[
    "specific_target",
    "unspecified_single_target",
    "multiple_targets",
    "workspace_scope",
    "unknown",
]

PlanInputKind = Literal[
    "dataset",
    "finding",
    "state",
    "user_request",
    "unknown",
]


class PlanInputRef(BaseModel):
    """计划步骤引用的输入对象。"""

    kind: PlanInputKind = Field(
        default="user_request",
        description="输入引用类型，例如 dataset / finding / state / user_request。",
    )
    ref: str = Field(
        default="",
        description="引用标识，例如数据集 alias、路径片段、finding 摘要或用户请求片段。",
    )
    role: str = Field(
        default="",
        description="该输入在本步骤中的作用。",
    )


class PlanStep(BaseModel):
    """一个自然语言执行步骤。"""

    id: str = Field(..., description="稳定步骤 ID，例如 step_1。")
    instruction: str = Field(..., description="可交给 Router/Agent 的自然语言任务。")
    purpose: str = Field(default="", description="本步骤的目的。")
    inputs: list[PlanInputRef] = Field(default_factory=list)
    expected_outputs: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)

    requires_user_input: bool = Field(
        default=False,
        description="该步骤是否需要用户补充信息后才能执行。",
    )
    user_question: str = Field(
        default="",
        description="如果 requires_user_input=true，需要问用户的问题。",
    )

    @field_validator("id", "instruction")
    @classmethod
    def _not_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("field cannot be empty")
        return value


class TaskPlan(BaseModel):
    """用户输入对应的执行计划。"""

    mode: PlanMode = "single_task"
    interpretation: InputInterpretation = "single_operation_ready"
    target_scope: TargetScope = "unknown"
    objective: str = ""
    requires_confirmation: bool = False
    steps: list[PlanStep] = Field(default_factory=list)

    @property
    def is_multi_step(self) -> bool:
        return len(self.steps) > 1

    @property
    def should_execute_directly(self) -> bool:
        return not self.requires_confirmation


_PLANNER_SYSTEM_PROMPT = """你是 Scarecrow 的 Task Planner。

你的职责：
- 把用户请求解释成可顺序执行的计划
- 判断 mode: single_task / explicit_multi_task / planned_task
- 判断 interpretation 和 target_scope
- 输出严格 JSON
- 不回答用户问题
- 不选择工具
- 不输出 capability
- 不写代码
- 不执行任务
- 不输出 markdown
- 不输出 ```json 代码块

你必须输出如下 JSON 结构：

{
  "mode": "single_task | explicit_multi_task | planned_task",
  "interpretation": "single_operation_ready | single_operation_missing_input | explicit_multi_task | open_ended_goal | collection_goal",
  "target_scope": "specific_target | unspecified_single_target | multiple_targets | workspace_scope | unknown",
  "objective": "用户目标的简短概括",
  "requires_confirmation": false,
  "steps": [
    {
      "id": "step_1",
      "instruction": "可交给 Router/Agent 的自然语言任务",
      "purpose": "本步骤的目的",
      "inputs": [
        {
          "kind": "dataset | finding | state | user_request | unknown",
          "ref": "输入引用标识",
          "role": "该输入在本步骤中的作用"
        }
      ],
      "expected_outputs": ["本步骤期望产生的结果"],
      "depends_on": [],
      "requires_user_input": false,
      "user_question": ""
    }
  ]
}

mode 定义：

1. single_task
用户请求是一个单一操作目标。
即使这个单一操作缺少必要参数，也仍然是 single_task，而不是 planned_task。
如果缺少必要参数，应使用 interpretation="single_operation_missing_input"，
并让唯一 step 的 requires_user_input=true。

2. explicit_multi_task
用户显式给出多个独立任务。
例如多行任务、编号任务、明确的先后任务。
这种模式下，不要创造用户没有要求的新任务，只整理和规范用户已经表达的任务。

3. planned_task
用户提出开放式复杂目标或集合式目标，需要 Planner 主动设计只读分析步骤。
planned_task 不是用来处理“单步请求缺少参数”的。
如果只是缺少文件名、字段名、路径或目标对象，使用 single_task + requires_user_input=true。

interpretation 定义：

1. single_operation_ready
用户请求是一个单步操作，并且目标对象足够明确，可以直接执行。

2. single_operation_missing_input
用户请求是一个单步操作，但缺少必要参数，例如目标文件、字段、路径或对象。
这种情况必须保持 mode="single_task"。
唯一 step 应设置 requires_user_input=true，并在 user_question 中询问用户补充什么。

3. explicit_multi_task
用户显式列出了多个任务，例如多行任务、编号任务、明确的先后任务。

4. open_ended_goal
用户提出开放式复杂目标，需要 Planner 主动设计步骤，例如完整分析、生成报告、端到端检查。

5. collection_goal
用户目标指向一个集合范围，例如当前工作区、当前工程、当前项目、全部数据或一组候选对象。
可以规划发现候选对象并逐个处理。

target_scope 定义：

1. specific_target
用户已经明确给出目标对象，例如具体文件、路径、字段或实体。

2. unspecified_single_target
用户想操作一个单一对象，但没有说明具体是哪一个。
这种情况不要规划发现+选择流程，除非用户明确要求先列候选。
应输出 mode="single_task"，interpretation="single_operation_missing_input"，
并让 step.requires_user_input=true。

3. multiple_targets
用户明确要求处理多个目标。

4. workspace_scope
用户目标范围是当前工作区、当前工程、当前项目、全部数据或一组候选对象。

5. unknown
无法判断目标范围。

计划步骤原则：

- 每个 step 的 instruction 必须是自然语言，不是工具调用。
- 每个 step 不得包含 tool name、capability、Python 代码。
- 每个 step 应该能被单独交给 Router/Agent。
- 如果 step 依赖已有任务状态中的数据集、发现或上下文，必须在 inputs 中声明。
- 如果目标对象不明确，并且当前任务状态也无法消解，使用 kind="unknown"，不要编造文件名。
- 不要编造用户没有提供、任务状态也没有支持的数据文件、字段或业务实体。
- 如果任务涉及写文件、删除文件、覆盖文件、联网、发送结果、访问密钥，requires_confirmation 必须为 true。
- 只读分析、预览、统计、总结通常不需要确认。
- depends_on 只能引用前面已经出现的 step id。
- expected_outputs 写本步骤应产生的结果类型，不要写最终答案。

可执行性原则：

- 每个 step 必须判断自己是否可以直接执行。
- 如果 step 缺少必要的数据文件、字段、路径或用户选择，并且不能通过当前任务状态或前置步骤获得，则 requires_user_input=true。
- 如果 requires_user_input=true，必须在 user_question 中写清楚需要用户补充什么。
- 如果 step 的 inputs 中包含 kind="unknown"，通常 requires_user_input=true。
- 但是，如果该 step 的目的就是发现或列出候选对象，则不需要用户输入。
- 不要让 Agent 随机选择一个文件、字段或对象。
- 如果用户明确授权“全部分析”“都看一下”“任选一个”，才可以不要求用户选择。

数据源处理原则：

- 如果用户说的是一个单一操作，但没有给出具体文件或对象：
  - 不要升级为 planned_task。
  - 输出 single_task。
  - 设置 interpretation="single_operation_missing_input"。
  - 设置 target_scope="unspecified_single_target"。
  - 唯一 step 设置 requires_user_input=true。
- 如果用户目标是当前工作区、当前工程、当前项目、所有数据、全部数据或一组候选对象：
  - 可以输出 planned_task。
  - 设置 interpretation="collection_goal" 或 "open_ended_goal"。
  - 设置 target_scope="workspace_scope" 或 "multiple_targets"。
  - 可以规划发现候选数据源、逐个预览、逐个分析等只读步骤。
- 如果当前任务状态中已经有明确数据集：
  - 可以在 inputs 中引用已有 dataset。
  - 不要重新要求用户提供已经存在于任务状态中的数据源。

当前任务状态只能用于理解上下文和引用已有对象，不要把它当成本轮工具结果。
"""


def plan_user_input(
    user_input: str,
    cfg: LLMConfig,
    task_state_brief: str | None = None,
) -> TaskPlan:
    """把用户输入规划成 TaskPlan。"""

    cleaned = user_input.strip()

    if not cleaned:
        return TaskPlan(
            mode="single_task",
            interpretation="single_operation_missing_input",
            target_scope="unknown",
            objective="",
            requires_confirmation=False,
            steps=[],
        )

    model = load_chat_model_from_config(cfg)
    state_block = task_state_brief or "暂无结构化任务状态。"

    user_prompt = f"""当前任务状态：

{state_block}

用户输入：

{cleaned}

请把用户输入解释成 TaskPlan。
输出严格 JSON。
"""

    try:
        msg = model.invoke(
            [
                SystemMessage(content=_PLANNER_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]
        )
        content = _message_content_to_text(msg.content)
        data = _extract_json_object(content)
        plan = TaskPlan.model_validate(data)
    except Exception:
        return _fallback_single_step_plan(cleaned)

    return _normalize_plan(plan, cleaned)


def _normalize_plan(plan: TaskPlan, original_input: str) -> TaskPlan:
    """清理 Planner 输出，保证 plan 可用。

    这里只做结构清理和一致性修正，不基于用户文本做关键词规则匹配。
    """

    steps: list[PlanStep] = []
    seen_ids: set[str] = set()

    for index, step in enumerate(plan.steps, start=1):
        instruction = step.instruction.strip()

        if not instruction:
            continue

        if instruction.startswith("/"):
            continue

        step_id = step.id.strip() or f"step_{index}"
        step_id = _normalize_step_id(step_id, fallback=f"step_{index}")

        if step_id in seen_ids:
            step_id = f"step_{index}"

        valid_depends_on = [
            dep for dep in step.depends_on
            if dep in seen_ids and dep != step_id
        ]

        normalized_inputs = _normalize_inputs(step.inputs)
        requires_user_input = step.requires_user_input or _has_unknown_input(
            normalized_inputs
        )

        seen_ids.add(step_id)

        steps.append(
            PlanStep(
                id=step_id,
                instruction=instruction,
                purpose=step.purpose.strip(),
                inputs=normalized_inputs,
                expected_outputs=[
                    item.strip()
                    for item in step.expected_outputs
                    if item.strip()
                ],
                depends_on=valid_depends_on,
                requires_user_input=requires_user_input,
                user_question=step.user_question.strip(),
            )
        )

    if not steps:
        return _fallback_single_step_plan(original_input)

    normalized = TaskPlan(
        mode=plan.mode,
        interpretation=plan.interpretation,
        target_scope=plan.target_scope,
        objective=plan.objective.strip() or original_input,
        requires_confirmation=plan.requires_confirmation,
        steps=steps,
    )

    return _enforce_plan_consistency(normalized, original_input)


def _enforce_plan_consistency(plan: TaskPlan, original_input: str) -> TaskPlan:
    """根据 Planner 输出的结构化语义字段做一致性修正。

    注意：
    - 这里不扫描用户原文关键词。
    - 只根据 mode / interpretation / target_scope / step flags 做结构修正。
    """

    if plan.interpretation == "single_operation_missing_input":
        step = _select_user_input_step(plan.steps)

        if step is None:
            return _fallback_missing_input_plan(original_input)

        step = PlanStep(
            id="step_1",
            instruction=step.instruction,
            purpose=step.purpose or "完成用户请求，但需要用户补充必要信息。",
            inputs=_ensure_unknown_input(step.inputs),
            expected_outputs=step.expected_outputs or ["完成该请求"],
            depends_on=[],
            requires_user_input=True,
            user_question=step.user_question
            or "请补充该任务所需的具体对象、文件名、字段或路径。",
        )

        return TaskPlan(
            mode="single_task",
            interpretation="single_operation_missing_input",
            target_scope=plan.target_scope
            if plan.target_scope != "unknown"
            else "unspecified_single_target",
            objective=plan.objective or original_input,
            requires_confirmation=plan.requires_confirmation,
            steps=[step],
        )

    if len(plan.steps) == 1 and plan.mode != "single_task":
        return TaskPlan(
            mode="single_task",
            interpretation=plan.interpretation,
            target_scope=plan.target_scope,
            objective=plan.objective or original_input,
            requires_confirmation=plan.requires_confirmation,
            steps=plan.steps,
        )

    return plan


def _select_user_input_step(steps: list[PlanStep]) -> PlanStep | None:
    """从计划中选择最适合代表“需要用户补充输入”的步骤。

    这是结构选择，不读取用户原文。
    """

    for step in steps:
        if step.requires_user_input:
            return step

    for step in steps:
        if _has_unknown_input(step.inputs):
            return step

    if steps:
        return steps[0]

    return None


def _ensure_unknown_input(inputs: list[PlanInputRef]) -> list[PlanInputRef]:
    """确保 step 至少包含一个 unknown input。"""

    if any(item.kind == "unknown" for item in inputs):
        return inputs

    return [
        *inputs,
        PlanInputRef(
            kind="unknown",
            ref="未指定的必要输入",
            role="需要用户补充后才能执行。",
        ),
    ]


def _fallback_missing_input_plan(user_input: str) -> TaskPlan:
    return TaskPlan(
        mode="single_task",
        interpretation="single_operation_missing_input",
        target_scope="unspecified_single_target",
        objective=user_input,
        requires_confirmation=False,
        steps=[
            PlanStep(
                id="step_1",
                instruction=user_input,
                purpose="完成用户请求，但缺少必要输入。",
                inputs=[
                    PlanInputRef(
                        kind="unknown",
                        ref="未指定的必要输入",
                        role="需要用户补充后才能执行。",
                    )
                ],
                expected_outputs=["完成该请求"],
                depends_on=[],
                requires_user_input=True,
                user_question="请补充该任务所需的具体对象、文件名、字段或路径。",
            )
        ],
    )


def _normalize_inputs(inputs: list[PlanInputRef]) -> list[PlanInputRef]:
    normalized: list[PlanInputRef] = []

    for item in inputs:
        ref = item.ref.strip()
        role = item.role.strip()

        if not ref and item.kind != "state":
            continue

        normalized.append(
            PlanInputRef(
                kind=item.kind,
                ref=ref,
                role=role,
            )
        )

    return normalized


def _has_unknown_input(inputs: list[PlanInputRef]) -> bool:
    """结构性判断：只要存在 unknown input，该步骤默认需要用户补充。"""

    return any(item.kind == "unknown" for item in inputs)


def _normalize_step_id(step_id: str, fallback: str) -> str:
    cleaned = step_id.strip().lower()
    cleaned = re.sub(r"[^a-z0-9_\-]+", "_", cleaned)
    cleaned = cleaned.strip("_-")
    return cleaned or fallback


def _fallback_single_step_plan(user_input: str) -> TaskPlan:
    return TaskPlan(
        mode="single_task",
        interpretation="single_operation_ready",
        target_scope="unknown",
        objective=user_input,
        requires_confirmation=False,
        steps=[
            PlanStep(
                id="step_1",
                instruction=user_input,
                purpose="完成用户请求。",
                inputs=[
                    PlanInputRef(
                        kind="user_request",
                        ref=user_input,
                        role="原始用户请求。",
                    )
                ],
                expected_outputs=["完成该请求"],
                depends_on=[],
                requires_user_input=False,
                user_question="",
            )
        ],
    )


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
    """从模型输出中提取 JSON object。"""

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
        raise ValueError(f"No JSON object found in planner output: {text}")

    data = json.loads(match.group(0))

    if not isinstance(data, dict):
        raise ValueError("Planner output JSON is not an object.")

    return data