"""LocalStorage — the filesystem under one root. Paths are root-relative."""

from __future__ import annotations

from pathlib import Path


class LocalStorage:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _resolve(self, path: str) -> Path:
        resolved = (self.root / path).resolve()
        if not resolved.is_relative_to(self.root.resolve()):
            raise ValueError(f"path {path!r} escapes the storage root {self.root}")
        return resolved

    def exists(self, path: str) -> bool:
        return self._resolve(path).is_file()

    def read_bytes(self, path: str) -> bytes:
        return self._resolve(path).read_bytes()

    def write_bytes(self, path: str, data: bytes) -> None:
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    def listdir(self, prefix: str) -> tuple[str, ...]:
        base = self._resolve(prefix) if prefix else self.root.resolve()
        if not base.is_dir():
            return ()
        found = (p for p in base.rglob("*") if p.is_file())
        return tuple(sorted(str(p.relative_to(self.root.resolve())) for p in found))

    def commit(self) -> None:
        """Already durable; Modal volumes are the ones that need a commit."""
