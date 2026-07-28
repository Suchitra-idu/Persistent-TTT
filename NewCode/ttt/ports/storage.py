"""Storage — a flat byte store. Checkpoints go through this, not through open()."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Storage(Protocol):
    def exists(self, path: str) -> bool: ...

    def read_bytes(self, path: str) -> bytes:
        """Raises FileNotFoundError for an absent path."""

    def write_bytes(self, path: str, data: bytes) -> None:
        """Creates intermediate directories; overwrites."""

    def listdir(self, prefix: str) -> tuple[str, ...]:
        """Paths under `prefix`, sorted. Absent prefixes list empty."""

    def commit(self) -> None:
        """Make prior writes durable. A no-op where writes already are."""
