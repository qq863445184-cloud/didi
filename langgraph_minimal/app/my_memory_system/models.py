from __future__ import annotations

from dataclasses import dataclass, field
from time import time
from typing import Any
from uuid import uuid4


@dataclass
class MemoryRecord:
    """A normalized memory item used by all memory stores.

    后续不管底层是内存、Qdrant 还是 Neo4j，都先统一成这个结构。
    这样 Agent 和 Tool 层不用关心具体存储实现。
    """

    content: str
    memory_type: str = "working"
    importance: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)
    memory_id: str = field(default_factory=lambda: str(uuid4()))
    created_at: float = field(default_factory=time)
    # 访问强化：被检索命中的次数与最近一次命中时间。
    # 模拟“记忆重放”——常被用到的记忆会被强化，更不容易被遗忘。
    access_count: int = 0
    last_accessed: float = 0.0

    def reinforce(self, now: float | None = None) -> None:
        """Mark this record as accessed once, strengthening it."""

        self.access_count += 1
        self.last_accessed = now if now is not None else time()


@dataclass
class MemorySearchResult:
    """A scored search hit returned by a memory store."""

    record: MemoryRecord
    score: float
    source: str

