"""ModalStorage — a mounted Modal Volume: LocalStorage plus `commit`, without which writes stay invisible."""

from __future__ import annotations

from pathlib import Path

from ttt.adapters.local_storage import LocalStorage


class ModalStorage:
    def __init__(self, volume, root: str | Path) -> None:
        self.volume = volume
        self._files = LocalStorage(root)

    def exists(self, path: str) -> bool:
        return self._files.exists(path)

    def read_bytes(self, path: str) -> bytes:
        return self._files.read_bytes(path)

    def write_bytes(self, path: str, data: bytes) -> None:
        self._files.write_bytes(path, data)

    def listdir(self, prefix: str) -> tuple[str, ...]:
        return self._files.listdir(prefix)

    def commit(self) -> None:
        self.volume.commit()
