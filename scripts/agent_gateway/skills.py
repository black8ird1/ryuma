"""Skill auto-discovery for the shared transport — backend-neutral.

A "skill" is a folder containing a SKILL.md whose YAML frontmatter carries a
`name` and `description` (the Claude Code convention, but any repo can adopt it).
The /skills menu lists them as tappable buttons; activating one submits a turn
that tells the backend's model to read and follow that skill's file. Backends
with a native skill tool (Claude) use it; others simply read the file. Same UX
for every backend, which is the point of the universal layer.

The skills directory is configurable (AGENT_GATEWAY_SKILLS_DIR, default
.claude/skills) so this is not bound to any one project.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from .core import ROOT

_NAME_RE = re.compile(r"^name:\s*(.+?)\s*$", re.MULTILINE)
_DESC_RE = re.compile(r"^description:\s*(.+?)\s*$", re.MULTILINE)


def skills_dir() -> Path:
    raw = os.environ.get("AGENT_GATEWAY_SKILLS_DIR", ".claude/skills")
    path = Path(raw)
    return path if path.is_absolute() else (ROOT / path)


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    path: str  # repo-relative path to SKILL.md (resolves inside agent worktrees too)


def _frontmatter(text: str) -> str:
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            return text[3:end]
    return text[:1000]


def _clean(value: str) -> str:
    return value.strip().strip('"').strip("'").strip()


def discover_skills(base: Path | None = None, *, repo: Path = ROOT, limit: int = 40) -> list[Skill]:
    base = base or skills_dir()
    skills: list[Skill] = []
    if not base.is_dir():
        return skills
    for sub in sorted(base.iterdir()):
        md = sub / "SKILL.md"
        if not md.is_file():
            continue
        try:
            head = md.read_text(encoding="utf-8", errors="replace")[:4000]
        except OSError:
            continue
        fm = _frontmatter(head)
        name_m = _NAME_RE.search(fm)
        name = _clean(name_m.group(1)) if name_m else sub.name
        desc_m = _DESC_RE.search(fm)
        desc = _clean(desc_m.group(1)) if desc_m else ""
        try:
            rel = str(md.relative_to(repo))
        except ValueError:
            rel = str(md)
        skills.append(Skill(name=name, description=desc, path=rel))
        if len(skills) >= limit:
            break
    return skills


def activation_prompt(skill: Skill) -> str:
    desc = f" ({skill.description[:160]})" if skill.description else ""
    return (
        f"Activate the '{skill.name}' skill{desc}.\n"
        f"Read {skill.path} and follow its instructions for this request."
    )
