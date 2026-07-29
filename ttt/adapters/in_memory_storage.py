"""InMemoryStorage — a dict of bytes. Checkpoint tests need no tmpdir."""

from __future__ import annotations


class InMemoryStorage:
    def __init__(self) -> None:
        self._blobs: dict[str, bytes] = {}
        self.commits = 0

    def exists(self, path: str) -> bool:
        return path in self._blobs

    def read_bytes(self, path: str) -> bytes:
        if path not in self._blobs:
            raise FileNotFoundError(path)
        return self._blobs[path]

    def write_bytes(self, path: str, data: bytes) -> None:
        self._blobs[path] = bytes(data)

    def listdir(self, prefix: str) -> tuple[str, ...]:
        prefix = prefix.rstrip("/") + "/" if prefix else ""
        return tuple(sorted(p for p in self._blobs if p.startswith(prefix)))

    def commit(self) -> None:
        self.commits += 1
