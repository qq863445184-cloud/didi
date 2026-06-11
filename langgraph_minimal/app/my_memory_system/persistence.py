from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import MemoryRecord


def record_to_dict(record: MemoryRecord) -> dict[str, Any]:
    """Serialize a MemoryRecord into JSON-friendly data."""

    return {
        "content": record.content,
        "memory_type": record.memory_type,
        "importance": record.importance,
        "metadata": record.metadata,
        "memory_id": record.memory_id,
        "created_at": record.created_at,
        "access_count": record.access_count,
        "last_accessed": record.last_accessed,
    }


def record_from_dict(data: dict[str, Any]) -> MemoryRecord:
    """Restore a MemoryRecord while tolerating older persisted payloads."""

    record = MemoryRecord(
        content=str(data.get("content", "")),
        memory_type=str(data.get("memory_type", "working")),
        importance=float(data.get("importance", 0.5)),
        metadata=dict(data.get("metadata") or {}),
        created_at=float(data.get("created_at", 0.0) or 0.0),
        access_count=int(data.get("access_count", 0) or 0),
        last_accessed=float(data.get("last_accessed", 0.0) or 0.0),
    )
    if data.get("memory_id"):
        record.memory_id = str(data["memory_id"])
    return record


class JsonMemoryPersistence:
    """Tiny JSON persistence for local stores.

    It keeps working/perceptual memories useful across process restarts without
    introducing SQLite or another service into the teaching example.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> list[MemoryRecord]:
        if not self.path.exists():
            return []
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return []
        rows = payload if isinstance(payload, list) else payload.get("records", [])
        records: list[MemoryRecord] = []
        for row in rows:
            if isinstance(row, dict) and row.get("content"):
                records.append(record_from_dict(row))
        return records

    def save(self, records: list[MemoryRecord]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = [record_to_dict(record) for record in records]
        self.path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
