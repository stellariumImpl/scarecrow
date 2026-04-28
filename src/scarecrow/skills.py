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


def build_system_prompt(skills_dir: Path, workspace: Path | None = None) -> str:
    """组装完整 system prompt:基础人设 + 工作区简报 + Skills 清单"""
    parts: list[str] = [BASE_PROMPT]

    if workspace is not None:
        from scarecrow.workspace import workspace_brief

        parts.append(workspace_brief(workspace))

    blocks = load_skill_blocks(skills_dir)
    if blocks:
        parts.append("--- 可用能力 ---\n\n" + "\n\n".join(blocks))

    return "\n\n".join(parts)


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

_BUILTIN_DATA_EXPLORER_SKILL = """---
name: data-explorer
description: 系统化探查 DataFrame:形状、dtype、缺失值、数值列分布、类别列基数。当用户说"看看这份数据"、"分析一下"、"探索"、"了解一下数据"、"数据长什么样"时使用
---

# data-explorer

## 何时使用

用户提出探查类、概览类请求时,按本 Skill 的工作流执行,而不是临场想分析步骤。

触发表达示例:
- "看看 xxx.csv 这份数据"
- "帮我分析一下这个文件"
- "了解一下数据情况"
- "探索 dataset/yyy.csv"
- "这数据长什么样"

**不**触发的场景:
- 用户已经指定具体操作("看前 5 行" / "统计 xx 列均值"),按用户说的做即可
- 用户问的是某个具体列/具体值,不需要全量探查

## 工作流(标准 4 步)

执行下列步骤,**用一次 `run_python` 调用做完**,不要拆成多次:

```python
# Step 1: 加载并保留为 df_<short_name>,后续可复用
df = pd.read_csv('path/to/file.csv')
print(f"=== Shape: {df.shape} ===\\n")

# Step 2: dtype 与缺失值
print("=== Dtypes & Missing ===")
info = pd.DataFrame({
    'dtype': df.dtypes.astype(str),
    'missing': df.isna().sum(),
    'missing_pct': (df.isna().sum() / len(df) * 100).round(2),
})
print(info.to_string())
print()

# Step 3: 数值列分布(只对数值列做 describe)
num_cols = df.select_dtypes(include='number').columns
if len(num_cols) > 0:
    print("=== Numeric Distribution ===")
    print(df[num_cols].describe().T[['mean', 'std', 'min', '50%', 'max']].round(3).to_string())
    print()

# Step 4: 类别列基数(object/category 列的 unique 数与 top 值)
cat_cols = df.select_dtypes(include=['object', 'category']).columns
if len(cat_cols) > 0:
    print("=== Categorical Cardinality ===")
    for col in cat_cols[:10]:  # 最多看 10 列,防爆
        nunique = df[col].nunique()
        top = df[col].value_counts().head(3).to_dict()
        print(f"{col}: {nunique} unique | top3: {top}")
```

## 大文件保护

数据 > 500 万行时,**先采样**:

```python
df_full = pd.read_csv('big_file.csv')
df = df_full.sample(n=100_000, random_state=42)  # 探查用采样
print(f"完整数据: {df_full.shape},探查使用 10 万行采样")
# 之后所有探查在 df 上做
```

向用户说明"以上是基于 10 万行采样的统计,趋势可信但具体数值仅供参考"。

## 输出原则

- 一次 `run_python` 调用拿到全部 4 步的输出
- 输出后用人类可读的方式总结:**这份数据的形状、有几个数值列、有几个类别列、有没有缺失、有没有异常**
- **不要**把 dataframe 整个 print 出来,也不要列所有 unique 值
- 总结后主动问用户下一步:"想看哪些列的细节 / 想做什么分析?"
"""


def ensure_builtin_skills(skills_dir: Path) -> None:
    """首次启动时把内置 Skills 写入 skills_dir。

    已有非空文件则不覆盖(尊重用户编辑),
    空文件视为"未初始化",重新写入。
    """
    builtins = {
        "run-python": _BUILTIN_RUN_PYTHON_SKILL,
        "data-explorer": _BUILTIN_DATA_EXPLORER_SKILL,
    }
    for name, content in builtins.items():
        target = skills_dir / name / "SKILL.md"
        if target.exists() and target.stat().st_size > 0:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
