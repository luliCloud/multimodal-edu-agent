"""Video storage behind a small interface so a cloud bucket can replace the local directory.

Keys are POSIX-style relative paths such as ``<project_id>/<render_id>/final.mp4``. The API
only ever hands out ``url(key)``; a cloud implementation can return a signed URL instead.
"""

import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

VIDEO_SUFFIXES = {".mp4", ".webm", ".mov", ".m4v"}
_KEY_PART = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._ -]*$")


@dataclass(frozen=True)
class StoredObject:
    key: str
    size_bytes: int
    modified_at: datetime


class VideoStorage(Protocol):
    name: str

    def put_file(self, key: str, source: Path, *, move: bool = False) -> StoredObject: ...
    def url(self, key: str) -> str: ...
    def local_path(self, key: str) -> Path | None: ...
    def list(self, prefix: str = "") -> list[StoredObject]: ...
    def delete_prefix(self, prefix: str) -> None: ...


def validate_key(key: str) -> str:
    parts = key.split("/")
    if not parts or any(part in ("", ".", "..") or not _KEY_PART.match(part) for part in parts):
        raise ValueError(f"Invalid storage key: {key!r}")
    return key


class LocalDirectoryStorage:
    """Stores videos as plain files under one root directory."""

    name = "local"

    def __init__(self, root: Path, url_prefix: str = "/api/media") -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.url_prefix = url_prefix.rstrip("/")

    def _resolve(self, key: str) -> Path:
        path = (self.root / validate_key(key)).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f"Invalid storage key: {key!r}")
        return path

    def put_file(self, key: str, source: Path, *, move: bool = False) -> StoredObject:
        target = self._resolve(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        (shutil.move if move else shutil.copy2)(source, target)
        return self._stat(key, target)

    def url(self, key: str) -> str:
        return f"{self.url_prefix}/{validate_key(key)}"

    def local_path(self, key: str) -> Path | None:
        try:
            path = self._resolve(key)
        except ValueError:
            return None
        return path if path.is_file() else None

    def list(self, prefix: str = "") -> list[StoredObject]:
        base = self._resolve(prefix) if prefix else self.root
        if not base.exists():
            return []
        objects = []
        for path in base.rglob("*"):
            if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
                key = path.relative_to(self.root).as_posix()
                try:
                    validate_key(key)
                except ValueError:
                    continue
                objects.append(self._stat(key, path))
        return sorted(objects, key=lambda item: item.modified_at, reverse=True)

    def delete_prefix(self, prefix: str) -> None:
        path = self._resolve(prefix)
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        elif path.is_file():
            path.unlink(missing_ok=True)

    @staticmethod
    def _stat(key: str, path: Path) -> StoredObject:
        stat = path.stat()
        return StoredObject(
            key=key, size_bytes=stat.st_size,
            modified_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
        )
