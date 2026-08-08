"""FakeWikiLangSource — rows scripted per language code, no network."""

from __future__ import annotations

from typing import Mapping, Sequence


class FakeWikiLangSource:
    def __init__(self, rows_by_code: Mapping[str, Sequence[dict]]) -> None:
        self._rows_by_code = {code: list(rows) for code, rows in rows_by_code.items()}
        self.calls: list[tuple[str, str, int]] = []

    def load(self, code: str, snapshot: str, limit: int, *, min_chars: int = 200) -> list[dict]:
        self.calls.append((code, snapshot, limit))
        available = self._rows_by_code.get(code, [])
        long_enough = [row for row in available if len(row["text"]) >= min_chars]
        return long_enough[:limit]
