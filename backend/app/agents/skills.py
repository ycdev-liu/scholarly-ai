"""Repository-owned Agent Skills, loaded in two stages."""

from dataclasses import dataclass
from pathlib import Path

import yaml

SKILLS_DIR = Path(__file__).resolve().parent / "skills"


@dataclass(frozen=True)
class SkillInfo:
    name: str
    description: str
    path: Path


def _parse_skill(path: Path) -> tuple[dict, str]:
    text = path.read_text(encoding="utf-8")
    parts = text.split("---", 2)
    if len(parts) != 3 or parts[0].strip():
        raise ValueError(f"Invalid SKILL.md frontmatter: {path}")
    metadata = yaml.safe_load(parts[1])
    if not isinstance(metadata, dict):
        raise ValueError(f"Invalid SKILL.md metadata: {path}")
    name, description = metadata.get("name"), metadata.get("description")
    if not isinstance(name, str) or name != path.parent.name or not name:
        raise ValueError(f"Invalid skill name: {path}")
    if not isinstance(description, str) or not description.strip():
        raise ValueError(f"Invalid skill description: {path}")
    return metadata, parts[2].strip()

# 罗列出技能skills 名字和描述
def discover_skills() -> dict[str, SkillInfo]:
    result = {}
    for path in sorted(SKILLS_DIR.glob("*/SKILL.md")):
        metadata, _ = _parse_skill(path)
        result[metadata["name"]] = SkillInfo(metadata["name"], metadata["description"], path)
    return result


def read_skill(name: str, registry: dict[str, SkillInfo]) -> str:
    if name not in registry:
        raise KeyError(f"Unknown skill: {name}")
    return _parse_skill(registry[name].path)[1]
