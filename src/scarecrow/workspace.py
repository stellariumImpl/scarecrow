"""工作区扫描：递归识别文件类型并统计数量（参考 datasage 风格）"""

from pathlib import Path

IGNORED_DIRS = {
    ".git", ".venv", "venv", "__pycache__", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", "node_modules", "dist", "build",
}

DATASET_EXTS = {".csv", ".parquet", ".jsonl"}
NOTEBOOK_EXTS = {".ipynb"}
SCRIPT_EXTS = {".py"}
CONFIG_NAMES = {"pyproject.toml", "requirements.txt", "environment.yml", "config.yaml"}

_OTHER_TEXT_SUFFIXES = {
    ".md", ".rst", ".txt",
    ".yaml", ".yml", ".json", ".toml", ".cfg", ".ini",
    ".sh", ".bash", ".sql",
    ".js", ".ts", ".jsx", ".tsx",
    ".html", ".css",
}

_OTHER_BARE_NAMES = {
    "Dockerfile", "Makefile", "README", "LICENSE",
    ".gitignore", ".dockerignore", ".env.example",
}


def should_ignore(path: Path) -> bool:
    return any(part in IGNORED_DIRS for part in path.parts)


def scan_workspace(root: Path) -> dict[str, int]:
    """递归扫描工作区，返回各类文件数量"""
    files = [
        p for p in root.rglob("*")
        if p.is_file() and not should_ignore(p.relative_to(root))
    ]

    counts = {"datasets": 0, "notebooks": 0, "scripts": 0, "configs": 0, "other": 0}

    for p in files:
        ext = p.suffix.lower()
        name = p.name

        if ext in DATASET_EXTS:
            counts["datasets"] += 1
        elif ext in NOTEBOOK_EXTS:
            counts["notebooks"] += 1
        elif ext in SCRIPT_EXTS:
            counts["scripts"] += 1
        elif name in CONFIG_NAMES:
            counts["configs"] += 1
        elif ext in _OTHER_TEXT_SUFFIXES or name in _OTHER_BARE_NAMES:
            counts["other"] += 1

    return counts


def workspace_summary(root: Path) -> str:
    """生成工作区摘要文字"""
    counts = scan_workspace(root)

    parts = [
        f"{counts['datasets']} datasets",
        f"{counts['notebooks']} notebooks",
        f"{counts['scripts']} scripts",
        f"{counts['configs']} configs",
        f"{counts['other']} other files",
    ]

    return " · ".join(parts)
