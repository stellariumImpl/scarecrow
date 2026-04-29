# src/scarecrow/workspace.py

"""工作区扫描与路径解析。"""

from pathlib import Path


_DATA_EXTS = {
    ".csv",
    ".parquet",
    ".jsonl",
    ".xlsx",
    ".xls",
}

_IGNORED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    "node_modules",
    ".idea",
    ".vscode",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "dist",
    "build",
    ".DS_Store",
}

_MAX_FILES = 50


def scan_data_files(workspace: Path) -> list[str]:
    """递归扫描工作区，返回相对路径形式的数据文件列表。"""

    if not workspace.exists() or not workspace.is_dir():
        return []

    found: list[str] = []

    for path in _walk(workspace):
        if path.suffix.lower() not in _DATA_EXTS:
            continue

        try:
            rel = path.relative_to(workspace)
        except ValueError:
            continue

        found.append(str(rel))

        if len(found) >= _MAX_FILES:
            break

    found.sort()
    return found


def scan_workspace_files(workspace: Path, max_depth: int = 5, limit: int = 500) -> list[str]:
    """扫描工作区文件，返回相对路径。

    用于路径解析，不读取文件内容。
    """

    if not workspace.exists() or not workspace.is_dir():
        return []

    found: list[str] = []

    for path in _walk(workspace):
        try:
            rel = path.relative_to(workspace)
        except ValueError:
            continue

        if len(rel.parts) > max_depth:
            continue

        found.append(str(rel))

        if len(found) >= limit:
            break

    found.sort()
    return found


def resolve_workspace_path(
    workspace: Path,
    query: str,
    data_files_only: bool = False,
    max_candidates: int = 10,
) -> list[str]:
    """根据用户输入解析工作区相对路径。

    返回候选相对路径列表，最匹配的排在前面。

    匹配策略：
    1. 完全相对路径匹配
    2. 文件名完全匹配
    3. 去扩展名文件名匹配
    4. 子串匹配
    """

    normalized_query = _normalize_path_query(query)

    if not normalized_query:
        return []

    candidates = (
        scan_data_files(workspace)
        if data_files_only
        else scan_workspace_files(workspace)
    )

    scored: list[tuple[int, str]] = []

    for candidate in candidates:
        score = _score_path_match(candidate, normalized_query)
        if score > 0:
            scored.append((score, candidate))

    scored.sort(key=lambda item: (-item[0], item[1]))

    return [path for _, path in scored[:max_candidates]]


def workspace_brief(workspace: Path) -> str:
    """生成可注入 system prompt 的工作区简报。"""

    files = scan_data_files(workspace)
    header = f"## 当前工作区\n\n路径: `{workspace}`"

    if not files:
        return (
            f"{header}\n\n"
            "工作区下未发现数据文件 "
            "(csv / parquet / jsonl / xlsx / xls)。"
        )

    truncated = len(files) >= _MAX_FILES
    file_lines = "\n".join(f"- `{file}`" for file in files)
    note = f"\n\n(已截断，只显示前 {_MAX_FILES} 个)" if truncated else ""

    return (
        f"{header}\n\n"
        f"数据文件清单（相对工作区路径，共 {len(files)} 个）:\n"
        f"{file_lines}"
        f"{note}\n\n"
        "用户提到文件名时，优先匹配以上清单。"
        "用户没明说路径时，直接使用清单中的相对路径，不要再 `os.listdir` 试探。"
    )


def _score_path_match(candidate: str, query: str) -> int:
    """给候选路径打分。"""

    candidate_norm = candidate.replace("\\", "/").lower()
    candidate_path = Path(candidate_norm)
    name = candidate_path.name
    stem = candidate_path.stem

    if candidate_norm == query:
        return 100

    if name == query:
        return 90

    if stem == query:
        return 80

    if candidate_norm.endswith("/" + query):
        return 70

    if query in name:
        return 60

    if query in stem:
        return 50

    if query in candidate_norm:
        return 40

    return 0


def _normalize_path_query(query: str) -> str:
    """规范化用户路径查询。"""

    cleaned = query.strip().strip("`'\"")
    cleaned = cleaned.replace("\\", "/")
    cleaned = cleaned.lower()

    # 常见自然语言里可能夹带的轻量噪声。
    for token in [
        "文件",
        "数据",
        "这个",
        "那个",
        "看一下",
        "看看",
        "分析",
        "探索",
    ]:
        cleaned = cleaned.replace(token, "")

    return cleaned.strip().strip("/")


def _walk(root: Path):
    """递归遍历目录，自动跳过 ignored 目录。"""

    try:
        entries = list(root.iterdir())
    except (PermissionError, OSError):
        return

    for entry in entries:
        if entry.name in _IGNORED_DIRS or entry.name.startswith("."):
            continue

        if entry.is_dir():
            yield from _walk(entry)
        elif entry.is_file():
            yield entry