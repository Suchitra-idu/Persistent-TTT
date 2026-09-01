"""InMemoryTracker — every logged dict, in order, for the app tests to assert on."""

from __future__ import annotations

from typing import Mapping


class InMemoryTracker:
    def __init__(self) -> None:
        self.records: list[dict[str, float]] = []
        self.images: list[tuple[str, str]] = []
        self.finished = False

    def log(self, metrics: Mapping[str, float]) -> None:
        self.records.append(dict(metrics))

    def log_image(self, key: str, path: str) -> None:
        self.images.append((key, path))

    def finish(self) -> None:
        self.finished = True

    def values_of(self, key: str) -> list[float]:
        return [record[key] for record in self.records if key in record]

    @property
    def keys_logged(self) -> frozenset[str]:
        return frozenset(key for record in self.records for key in record)
