"""WikipediaLangSource — one language config of wikimedia/wikipedia, streamed.

Streaming avoids downloading a whole config (Malagasy alone is ~100k rows)
when the prep step only ever needs a few hundred to a couple thousand.
"""

from __future__ import annotations


class WikipediaLangSource:
    def load(
        self, code: str, snapshot: str, limit: int, *, min_chars: int = 200
    ) -> list[dict]:
        from datasets import load_dataset

        ds = load_dataset(
            "wikimedia/wikipedia", f"{snapshot}.{code}", split="train", streaming=True
        )
        rows: list[dict] = []
        for row in ds:
            text = row.get("text", "")
            if len(text) < min_chars:
                continue
            rows.append({"text": text, "title": row.get("title", "")})
            if len(rows) >= limit:
                break
        return rows
