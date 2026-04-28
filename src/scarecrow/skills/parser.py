from pathlib import Path

from scarecrow.skills.schemas import SkillDocument, SkillMetadata, SkillSource


def parse_skill_file(path: Path, source: SkillSource = "user") -> SkillDocument | None:
    """解析单个 SKILL.md。

    支持简单 frontmatter：

    ---
    name: data-explorer
    description: 系统化探查 DataFrame
    ---

    # data-explorer
    ...
    """

    try:
        raw = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None

    if not raw:
        return None

    frontmatter, body = _split_frontmatter(raw)

    name = frontmatter.get("name") or path.parent.name
    description = frontmatter.get("description", "")

    metadata = SkillMetadata(
        name=name.strip(),
        description=description.strip(),
        path=path,
        source=source,
        enabled=True,
    )

    return SkillDocument(
        metadata=metadata,
        content=raw,
    )


def _split_frontmatter(raw: str) -> tuple[dict[str, str], str]:
    """解析最小 YAML-like frontmatter。

    为了避免现在引入 pyyaml，先只支持 key: value。
    后面 skill schema 复杂后再换成 yaml.safe_load。
    """

    if not raw.startswith("---"):
        return {}, raw

    lines = raw.splitlines()

    end_idx = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end_idx = i
            break

    if end_idx is None:
        return {}, raw

    fm_lines = lines[1:end_idx]
    body = "\n".join(lines[end_idx + 1 :]).strip()

    data: dict[str, str] = {}

    for line in fm_lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue

        key, value = line.split(":", 1)
        data[key.strip()] = value.strip().strip('"').strip("'")

    return data, body
