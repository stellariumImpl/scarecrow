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

## 错误处理原则

- **文件不存在时不要自救**：如果 `pd.read_csv('xxx.csv')` 报 `FileNotFoundError`，直接告诉用户"文件 xxx.csv 不存在，请确认路径"，**不要**主动 `os.listdir` 找替代品
- 只有用户明确说"帮我找数据" / "看看目录里有什么" / "扫描一下"时，才可以遍历目录
- 一次回答中工具调用次数尽量不超过 2 次，能一次解决就不要拆成多次
- 报错先看错误类型再决定怎么办，不要无脑重试
"""


def ensure_builtin_skills(skills_dir: Path) -> None:
    """首次启动时把内置 Skill 写入 skills_dir。

    已有非空文件则不覆盖（尊重用户编辑），
    但空文件会被视为"未初始化"，重新写入内置内容。
    """
    target = skills_dir / "run-python" / "SKILL.md"
    if target.exists() and target.stat().st_size > 0:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(_BUILTIN_RUN_PYTHON_SKILL, encoding="utf-8")