"""SquadQaSource — real question/context pairs for RULER's `qa` task."""

from __future__ import annotations

from ttt.core.ruler_types import QaPair

# The bare "squad" shortname is a dead alias — the Hub only serves the
# namespaced repo now.
DATASET = "rajpurkar/squad"


class SquadQaSource:
    def load(self, limit: int) -> tuple[QaPair, ...]:
        from datasets import load_dataset

        pairs: list[QaPair] = []
        for row in load_dataset(DATASET, split="train"):
            answers = tuple(row["answers"]["text"])
            if not answers:
                continue
            pairs.append(
                QaPair(query=row["question"], answers=answers, context=row["context"])
            )
            if len(pairs) >= limit:
                break
        return tuple(pairs)
