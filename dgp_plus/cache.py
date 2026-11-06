"""Resumable on-disk cache for the expensive stages.

Each artifact lives in <root>/<stage>/<key>/, where the key hashes everything that decides its content. A
metadata.json next to it records those fields. Summaries are append-only JSONL, so a stopped run resumes where it
stopped, and several shard processes can fill one store.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def artifact_dir(root: str | Path, stage: str, metadata: dict) -> Path:
    """Folder for one artifact, created with its metadata.json on first use."""
    key = hashlib.sha1(json.dumps(metadata, sort_keys=True, default=str).encode()).hexdigest()[:16]
    directory = Path(root) / stage / key
    directory.mkdir(parents=True, exist_ok=True)
    meta_path = directory / "metadata.json"
    if not meta_path.exists():
        meta_path.write_text(json.dumps({**metadata, "stage": stage, "key": key}, indent=2, default=str), encoding="utf-8")
    return directory


class TextStore:
    """Append-only key -> text store backed by a JSONL file."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.data: dict[str, str] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue                 # another writer's half-written last line; that key is regenerated
                self.data[row["key"]] = row["text"]

    def __contains__(self, key: str) -> bool:
        return key in self.data

    def __getitem__(self, key: str) -> str:
        return self.data[key]

    def __len__(self) -> int:
        return len(self.data)

    def add_many(self, rows: list[tuple[str, str]]) -> None:
        """One O_APPEND write per batch, so parallel shards never interleave partial lines."""
        payload = "".join(json.dumps({"key": k, "text": t}, ensure_ascii=False) + "\n" for k, t in rows).encode("utf-8")
        fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o664)
        try:
            view = memoryview(payload)
            while view:
                view = view[os.write(fd, view):]
        finally:
            os.close(fd)
        self.data.update(rows)
