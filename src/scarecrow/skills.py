# src/scarecrow/skills.py

"""Skills 扫描与注入：递归读取 ~/.scarecrow/skills 下的 SKILL.md，拼成 system prompt"""

from pathlib import Path

# 仅基础人设写死，其余能力描述全部来自 SKILL.md
BASE_PROMPT = "你是 Scarecrow，一个本地数据分析助手。用中文回答用户。"


def load_skill_blocks(skills_dir: Path) -> list[str]:
    """读取 skills_dir 下所有 SKILL.md 内容，返回字符串列表"""
    if not skills_dir.exists():
        return []

    blocks: list[str] = []
    for skill_md in sorted(skills_dir.rglob("SKILL.md")):
        try:
            content = skill_md.read_text(encoding="utf-8").strip()
            if content:
                blocks.append(content)
        except Exception:
            continue  # 单个 Skill 读取失败不阻塞整体
    return blocks


def build_system_prompt(skills_dir: Path) -> str:
    """组装完整 system prompt：基础人设 + Skills 清单"""
    blocks = load_skill_blocks(skills_dir)
    if not blocks:
        return BASE_PROMPT
    return BASE_PROMPT + "\n\n--- 可用能力 ---\n\n" + "\n\n".join(blocks)


_BUILTIN_RUN_PYTHON_SKILL = """---
name: run-python
description: 当用户需要查看、加载、分析、统计、清洗数据时使用
---

# run-python

## 何时使用

用户请求涉及以下情况时，调用 `run_python` 工具：

- 加载数据文件（csv / parquet / jsonl / xlsx）
- 查看数据形状、列名、dtype、统计摘要
- 过滤、聚合、分组、排序、连接
- 计算指标、检测异常值、清洗数据

## 关键约定

- 环境已预装：`pd` (pandas)、`np` (numpy)
- **变量跨轮保留**：第一次 `df = pd.read_csv(...)`，下一轮可以直接用 `df`，不必重新加载
- **想看到结果必须 print()**，函数返回值不会自动显示
- 大 DataFrame 用 `.head()` / `.describe()` / `.info()`，**不要** `print(df)` 全部内容
- 大文件用 `nrows=1000` 先采样

## 不能做的事

- 没有装 matplotlib / seaborn，**无法绘图**
- 没有装 requests，**不能联网**
- 不要使用 `input()`，会卡住

## 工作流

1. 用户给出文件名 → 先 `print(pd.read_csv(path).head())` 确认结构
2. 加载完成赋值给 `df`，后续操作复用
3. 出错先 print `df.dtypes` 和 `df.columns` 排查
"""


def ensure_builtin_skills(skills_dir: Path) -> None:
    """首次启动时把内置 Skill 写入 skills_dir（已存在则不覆盖）"""
    target = skills_dir / "run-python" / "SKILL.md"
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(_BUILTIN_RUN_PYTHON_SKILL, encoding="utf-8")