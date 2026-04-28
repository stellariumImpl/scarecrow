# Scarecrow

本地 CLI 数据分析助手。在你自己的电脑上跑一个 LLM Agent,用自然语言读 csv、查询、统计、清洗数据。

## 特点

- **本地 REPL**:一个命令启动,在终端直接对话
- **持久 Python 环境**:跨多轮对话保留 DataFrame 变量,不用反复加载
- **工作区感知**:启动时自动扫描数据文件,LLM 知道你目录里有什么
- **Skill 系统**:`~/.scarecrow/skills/` 下的 SKILL.md 即插即用,行为可定制
- **多 Provider**:支持 OpenAI / DeepSeek / Ollama,API key 本地存储
- **可选 LangSmith trace**:网页可视化每次推理与 tool call

## 安装

```bash
git clone https://github.com/stellariumImpl/scarecrow
cd scarecrow
uv tool install -e . --force
```

要求 Python ≥ 3.13。

## 配置

进入任一工作目录,启动:

```bash
scarecrow
```

首次使用配置 LLM:

```
Scarecrow > /config
```

按提示选择 provider、模型、填入 API key。配置保存在 `~/.scarecrow/config.json` (权限 600)。

## 使用

启动后直接对话:

```
Scarecrow > 看一下 dataset/user_features.csv 的前 5 行
Scarecrow > 帮我探索一下 user_features
Scarecrow > 数值列的相关性怎么样?
```

变量在多轮对话间保留:第一次 `df = pd.read_csv(...)` 后续可直接用 `df`。

## 命令

| 命令 | 说明 |
|---|---|
| `/help` | 显示帮助 |
| `/config` | 配置 LLM provider |
| `/langsmith` | 配置 LangSmith trace (可选) |
| `/reset` | 清空对话历史与 Python 命名空间 |
| `/quit` | 退出 |

## Skill 自定义

`~/.scarecrow/skills/` 下每个子目录就是一个 Skill,内含 `SKILL.md`。启动时自动扫描注入 system prompt。

内置 Skill:

- `run-python`:Python 执行规范、错误处理原则
- `data-explorer`:DataFrame 系统化探查工作流

加新能力:新建目录 `~/.scarecrow/skills/<name>/SKILL.md`,按下列格式写:

```markdown
---
name: my-skill
description: 何时使用这个 Skill 的简短描述
---

# my-skill

## 何时使用
...

## 工作流
...
```

重启即生效。

## 目录结构

```
~/.scarecrow/
    config.json       # LLM 配置
    history.txt       # REPL 历史
    skills/           # Skill 目录
        run-python/SKILL.md
        data-explorer/SKILL.md
```

## 限制

- `run_python` 直接 `exec()` 用户代码,**只在自己机器上用,不要做服务端部署**
- 默认未装 matplotlib / requests,不能绘图、不能联网
- 单次工具输出超过 8000 字符会被截断

## License

MIT