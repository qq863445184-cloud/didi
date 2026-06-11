from __future__ import annotations

from time import time
from typing import Any

from .models import MemoryRecord, MemorySearchResult
from .stores import MemoryStore, WorkingMemoryStore


class MyMemoryManager:
    """Dispatch memory actions to concrete stores.

    这一层负责“策略”：根据 memory_type 找对应 store。Tool 层只解析参数，
    Store 层只管存储和检索，中间的调度都放在 Manager，后续扩展更干净。
    """

    def __init__(
        self,
        user_id: str = "default",
        stores: dict[str, MemoryStore] | None = None,
    ) -> None:
        self.user_id = user_id
        self.stores: dict[str, MemoryStore] = stores or {
            "working": WorkingMemoryStore(),
        }
        self.trace_events: list[dict[str, Any]] = []

    def add(
        self,
        *,
        content: str,
        memory_type: str = "working",
        importance: float = 0.5,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryRecord:
        store = self._get_store(memory_type)
        record = MemoryRecord(
            content=content,
            memory_type=memory_type,
            importance=importance,
            metadata={
                "user_id": self.user_id,
                **(metadata or {}),
            },
        )
        saved = store.add(record)
        self.trace_events.append(
            {
                "stage": "manager.add",
                "memory_type": memory_type,
                "memory_id": saved.memory_id,
                "content": saved.content,
            }
        )
        return saved

    def search(
        self,
        *,
        query: str,
        memory_type: str = "working",
        limit: int = 5,
    ) -> list[MemorySearchResult]:
        store = self._get_store(memory_type)
        results = store.search(query, limit=limit)
        self.trace_events.append(
            {
                "stage": "manager.search",
                "memory_type": memory_type,
                "query": query,
                "limit": limit,
                "hits": len(results),
            }
        )
        return results

    def summary(self, *, memory_type: str = "working", limit: int = 5) -> list[MemoryRecord]:
        store = self._get_store(memory_type)
        records = store.summary(limit=limit)
        self.trace_events.append(
            {
                "stage": "manager.summary",
                "memory_type": memory_type,
                "limit": limit,
                "count": len(records),
            }
        )
        return records

    def forget(
        self,
        *,
        memory_type: str = "working",
        strategy: str = "importance",
        importance_threshold: float = 0.3,
        max_age_seconds: float | None = None,
        capacity: int | None = None,
        now: float | None = None,
    ) -> list[MemoryRecord]:
        """Drop records the agent no longer needs.

        模拟认知遗忘：低价值、过期或超容量的记忆被清除，让记忆系统保持精简。
        三种策略可单独使用：
        - importance: 删除 importance 低于阈值的记忆。
        - age:        删除存在时间超过 max_age_seconds 的记忆。
        - capacity:   只保留“重要性优先、其次新鲜”的前 capacity 条，其余删除。
        """

        store = self._get_store(memory_type)
        records = store.all_records()
        current_time = now if now is not None else time()

        if strategy == "importance":
            victims = [r for r in records if r.importance < importance_threshold]
        elif strategy == "age":
            if max_age_seconds is None:
                raise ValueError("strategy=age 需要提供 max_age_seconds")
            victims = [
                r for r in records if (current_time - r.created_at) > max_age_seconds
            ]
        elif strategy == "capacity":
            if capacity is None:
                raise ValueError("strategy=capacity 需要提供 capacity")
            ranked = sorted(
                records,
                key=lambda item: (item.importance, item.created_at),
                reverse=True,
            )
            victims = ranked[capacity:]
        else:
            raise ValueError(
                f"暂不支持 forget strategy={strategy}，可选: importance, age, capacity"
            )

        forgotten: list[MemoryRecord] = []
        for record in victims:
            if store.delete(record.memory_id):
                forgotten.append(record)

        self.trace_events.append(
            {
                "stage": "manager.forget",
                "memory_type": memory_type,
                "strategy": strategy,
                "scanned": len(records),
                "forgotten": len(forgotten),
            }
        )
        return forgotten

    def consolidate(
        self,
        *,
        source_type: str = "working",
        target_type: str = "semantic",
        importance_threshold: float = 0.7,
        delete_source: bool = True,
    ) -> list[MemoryRecord]:
        """Promote valuable short-term memories into long-term storage.

        模拟记忆整合：把工作记忆里重要性达标的记录写入长期记忆（默认语义记忆），
        默认整合后从源 store 删除，避免短期记忆无限堆积。
        """

        source = self._get_store(source_type)
        target = self._get_store(target_type)

        candidates = [
            r for r in source.all_records() if r.importance >= importance_threshold
        ]
        consolidated: list[MemoryRecord] = []
        for record in candidates:
            promoted = MemoryRecord(
                content=record.content,
                memory_type=target_type,
                importance=record.importance,
                metadata={
                    **record.metadata,
                    "consolidated_from": source_type,
                    "origin_memory_id": record.memory_id,
                },
            )
            target.add(promoted)
            consolidated.append(promoted)
            if delete_source:
                source.delete(record.memory_id)

        self.trace_events.append(
            {
                "stage": "manager.consolidate",
                "source_type": source_type,
                "target_type": target_type,
                "candidates": len(candidates),
                "consolidated": len(consolidated),
            }
        )
        return consolidated

    def _get_store(self, memory_type: str) -> MemoryStore:
        if memory_type not in self.stores:
            supported = ", ".join(sorted(self.stores))
            raise ValueError(f"暂不支持 memory_type={memory_type}，当前支持: {supported}")
        return self.stores[memory_type]
