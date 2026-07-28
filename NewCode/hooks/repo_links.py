"""Rewrite links that leave docs/ into repo URLs at build time.

`[scan](../ttt/core/ttt_math.py)` is clickable in an editor and 404s in a
built site. Rewriting at build time keeps the markdown editor-friendly without
duplicating source into docs_dir.
"""

from __future__ import annotations

import re
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parents[1] / "docs"
REPO_ROOT = Path(__file__).resolve().parents[2]
BLOB = "https://github.com/Suchitra-idu/Our-TTT/blob/main"

OUTBOUND_LINK = re.compile(r"\[([^\]]+)\]\((\.\./[^)]+)\)")


def _rewrite(match: re.Match) -> str:
    text, target = match.group(1), match.group(2)
    path, _, anchor = target.partition("#")
    resolved = (DOCS_DIR / path).resolve()
    try:
        relative = resolved.relative_to(REPO_ROOT)
    except ValueError:
        return match.group(0)
    suffix = f"#{anchor}" if anchor else ""
    return f"[{text}]({BLOB}/{relative}{suffix})"


def on_page_markdown(markdown: str, **_kwargs) -> str:
    return OUTBOUND_LINK.sub(_rewrite, markdown)
