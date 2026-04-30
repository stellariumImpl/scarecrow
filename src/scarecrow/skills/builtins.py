from pathlib import Path

BUILTIN_RUN_PYTHON_SKILL = """---
name: run-python
description: 当用户需要查看、加载、分析、统计、清洗数据时使用
capabilities: python.execute, data.analyze, data.aggregate, data.clean, data.transform
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
- **想看到结果必须用 `print()`**，函数返回值不会自动显示
- 大 DataFrame 用 `.head()` / `.describe()` / `.info()`，**不要** `print(df)` 全部内容
- 大文件用 `nrows=1000` 先采样
- 不要随意覆盖用户已有变量；如果需要新变量，优先使用语义化名称，例如 `users_df`、`orders_df`

## 不能做的事

- 没有装 matplotlib / seaborn，**不要假设可以绘图**
- 没有装 requests，**不要假设可以联网**
- 不要使用 `input()`，会卡住
- 不要执行危险系统命令
- 不要删除、覆盖、移动用户文件，除非用户明确要求

## 工作流

1. 用户给出文件名 → 先读取少量样本确认结构
2. 加载完成后赋值给稳定变量，后续操作复用
3. 出错先看错误类型，不要无脑重试
4. 若是字段名问题，先打印 `df.columns`
5. 若是类型问题，先打印 `df.dtypes`
6. 若是缺失值问题，先打印 `df.isna().sum()`

## 推荐代码风格

```python
df = pd.read_csv("path/to/file.csv")
print(df.head())
print(df.shape)
print(df.dtypes)
```

如果用户只要求查看前几行：

```python
df = pd.read_csv("path/to/file.csv")
print(df.head(5).to_string())
```

如果用户要求统计摘要：

```python
print(df.describe(include="all").T.to_string())
```

## 错误处理原则

- **文件不存在时不要自救**：如果 `pd.read_csv("xxx.csv")` 报 `FileNotFoundError`，直接告诉用户“文件 xxx.csv 不存在，请确认路径”，**不要**主动 `os.listdir` 找替代品
- 只有用户明确说“帮我找数据” / “看看目录里有什么” / “扫描一下”时，才可以遍历目录
- 一次回答中工具调用次数尽量不超过 2 次，能一次解决就不要拆成多次
- 报错先看错误类型再决定怎么办，不要无脑重试
- 如果输出过长，主动建议用户缩小范围，例如指定列、采样、过滤条件

## 输出原则

- 工具执行后，用人类可读语言总结结果
- 不要把原始工具输出机械重复一遍
- 如果发现异常、缺失、类型不合理，要明确指出
- 如果用户的需求不完整，可以先给出已能判断的结果，再提出下一步建议
"""


BUILTIN_DATA_EXPLORER_SKILL = """---
name: data-explorer
description: 系统化探查 DataFrame：形状、字段、类型、缺失值、数值分布、类别基数。当用户说“看看这份数据”、“分析一下”、“探索”、“了解一下数据”、“数据长什么样”时使用
capabilities: data.explore, data.profile, data.missing_analysis
---

# data-explorer

## 何时使用

用户提出探查类、概览类请求时，按本 Skill 的工作流执行，而不是临场随意分析。

触发表达示例：

- “看看 xxx.csv 这份数据”
- “帮我分析一下这个文件”
- “了解一下数据情况”
- “探索 dataset/yyy.csv”
- “这数据长什么样”
- “先做一下 EDA”
- “帮我看看这个数据有没有问题”

## 不触发的场景

以下情况不需要完整探查：

- 用户已经指定具体操作，例如“看前 5 行”
- 用户只问某个具体列，例如“age 的均值是多少”
- 用户只要求转换格式、清洗某个字段
- 用户只是在闲聊或问项目设计

## 标准工作流

执行下列步骤，优先用一次 `run_python` 调用完成，不要拆成很多次：

```python
df = pd.read_csv("path/to/file.csv")

print(f"=== Shape: {df.shape} ===")
print()

print("=== Head ===")
print(df.head().to_string())
print()

print("=== Columns ===")
print(list(df.columns))
print()

print("=== Dtypes & Missing ===")
info = pd.DataFrame({
    "dtype": df.dtypes.astype(str),
    "missing": df.isna().sum(),
    "missing_pct": (df.isna().sum() / len(df) * 100).round(2),
})
print(info.to_string())
print()

num_cols = df.select_dtypes(include="number").columns
if len(num_cols) > 0:
    print("=== Numeric Distribution ===")
    print(
        df[num_cols]
        .describe()
        .T[["mean", "std", "min", "50%", "max"]]
        .round(3)
        .to_string()
    )
    print()

cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns
if len(cat_cols) > 0:
    print("=== Categorical Cardinality ===")
    for col in cat_cols[:10]:
        nunique = df[col].nunique(dropna=True)
        top = df[col].value_counts(dropna=True).head(3).to_dict()
        print(f"{col}: {nunique} unique | top3: {top}")
```

## 大文件保护

如果数据很大，不要一开始就全量重操作。

推荐先采样：

```python
df = pd.read_csv("path/to/file.csv", nrows=100_000)
print(f"当前使用前 100000 行做快速探查: {df.shape}")
```

如果用户明确需要全量统计，再读取全量。

如果必须读取全量，先说明风险，并避免打印大量内容。

## 多文件探查

如果用户要求比较多个文件，先分别输出：

- 文件路径
- shape
- columns
- dtypes
- 缺失率最高的字段
- 可能的 join key

不要一开始就盲目 merge。

## 输出总结模板

工具输出后，最终回答应该总结：

1. 数据规模：多少行、多少列
2. 字段类型：数值列、类别列、时间列的大致情况
3. 缺失情况：是否有高缺失字段
4. 异常迹象：是否有明显极端值、类型异常、重复字段
5. 下一步建议：可以继续做什么分析

示例：

“这份数据有 10000 行、12 列，其中 7 个数值列、5 个类别列。`age` 和 `income` 有少量缺失，`user_id` 看起来可以作为用户标识。下一步可以继续看目标变量分布，或者检查重复用户。”

## 不要做的事

- 不要把所有 unique value 全部打印出来
- 不要对所有列做昂贵操作
- 不要默认绘图
- 不要默认删除缺失值
- 不要默认修改原始文件
- 不要在没有用户确认时保存清洗结果
"""


def ensure_builtin_skills(skills_dir: Path) -> None:
    """首次启动时把内置 Skills 写入 skills_dir。

    已有非空文件则不覆盖，尊重用户编辑。
    空文件视为未初始化，会被重新写入。
    """

    builtins = {
        "run-python": BUILTIN_RUN_PYTHON_SKILL,
        "data-explorer": BUILTIN_DATA_EXPLORER_SKILL,
    }

    for name, content in builtins.items():
        target = skills_dir / name / "SKILL.md"

        if target.exists() and target.stat().st_size > 0:
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
