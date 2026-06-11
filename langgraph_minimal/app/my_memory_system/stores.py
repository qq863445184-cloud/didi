from __future__ import annotations

import re
from collections import Counter
from time import time
from typing import Any, Protocol

from .models import MemoryRecord, MemorySearchResult
from .entity_extraction import SpacyEntityExtractor, normalize_entity_id
from .persistence import JsonMemoryPersistence, record_from_dict, record_to_dict
from .scoring import combined_score


class MemoryStore(Protocol):
    """Small interface each memory backend should implement."""

    def add(self, record: MemoryRecord) -> MemoryRecord:
        ...

    def search(self, query: str, limit: int = 5) -> list[MemorySearchResult]:
        ...

    def summary(self, limit: int = 5) -> list[MemoryRecord]:
        ...

    def all_records(self) -> list[MemoryRecord]:
        """Enumerate every record so 遗忘/整合 can scan and pick candidates."""
        ...

    def delete(self, memory_id: str) -> bool:
        """Remove one record by id; return False if it was not found."""
        ...

    def update(self, memory_id: str, **changes: Any) -> MemoryRecord | None:
        """Update one record by id; return None if it was not found."""
        ...

    def clear(self) -> int:
        """Remove all records from this store and return the number removed."""
        ...


class WorkingMemoryStore:
    """A lightweight in-process store for short-term memory.

    这个 store 不依赖外部服务，适合先把工具协议、Agent 调用和 trace 跑通。
    检索用关键词重叠作为基础分，再叠加统一的“重要性 + 时间近因”加权；
    后续 semantic memory 会换成 embedding + Qdrant，但评分口径保持一致。
    """

    def __init__(
        self,
        *,
        half_life_seconds: float = 3600.0,
        recency_weight: float = 0.3,
        access_gain: float = 0.5,
        persistence_path: str | None = None,
        ttl_seconds: float | None = None,
        max_records: int | None = None,
    ) -> None:
        self.persistence = (
            JsonMemoryPersistence(persistence_path) if persistence_path else None
        )
        self.records: list[MemoryRecord] = (
            self.persistence.load() if self.persistence is not None else []
        )
        # 工作记忆偏短期，默认 1 小时半衰、近因占基础分 30%。
        self.half_life_seconds = half_life_seconds
        self.recency_weight = recency_weight
        # 访问强化上限增益：被反复命中的记忆最多获得 (1 + access_gain) 倍加成。
        self.access_gain = access_gain
        # 可选自动维护策略：工作记忆可以像“短期缓存”一样按 TTL/容量自清理。
        self.ttl_seconds = ttl_seconds
        self.max_records = max_records

    def add(self, record: MemoryRecord) -> MemoryRecord:
        self.records.append(record)
        self.prune()
        self._persist()
        return record

    def add_record(
        self,
        *,
        content: str,
        memory_type: str = "working",
        importance: float = 0.5,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryRecord:
        """Convenience helper for tests and demos that do not need a Manager."""

        return self.add(
            MemoryRecord(
                content=content,
                memory_type=memory_type,
                importance=importance,
                metadata=metadata or {},
            )
        )

    def search(self, query: str, limit: int = 5) -> list[MemorySearchResult]:
        self.prune()
        query_terms = self._tokenize(query)
        now = time()
        scored: list[MemorySearchResult] = []

        for record in self.records:
            content_terms = self._tokenize(record.content)
            overlap = self._overlap(query_terms, content_terms)
            if overlap <= 0:
                continue
            score = combined_score(
                overlap,
                record.importance,
                created_at=record.created_at,
                now=now,
                half_life_seconds=self.half_life_seconds,
                recency_weight=self.recency_weight,
                access_count=record.access_count,
                access_gain=self.access_gain,
            )
            scored.append(
                MemorySearchResult(record=record, score=score, source="working")
            )

        scored.sort(key=lambda item: item.score, reverse=True)
        top = scored[:limit]
        # 检索即强化：本次返回的记忆被“想起”，下次更易被检索到。
        for item in top:
            item.record.reinforce(now)
        if top:
            self._persist()
        return top

    def summary(self, limit: int = 5) -> list[MemoryRecord]:
        self.prune()
        # 重要性优先，其次新记忆优先，便于短期工作记忆保持“当前感”。
        return sorted(
            self.records,
            key=lambda item: (item.importance, item.created_at),
            reverse=True,
        )[:limit]

    def all_records(self) -> list[MemoryRecord]:
        return list(self.records)

    def delete(self, memory_id: str) -> bool:
        for index, record in enumerate(self.records):
            if record.memory_id == memory_id:
                del self.records[index]
                self._persist()
                return True
        return False

    def update(self, memory_id: str, **changes: Any) -> MemoryRecord | None:
        for record in self.records:
            if record.memory_id != memory_id:
                continue
            self._apply_record_changes(record, changes)
            self._persist()
            return record
        return None

    def clear(self) -> int:
        count = len(self.records)
        self.records = []
        self._persist()
        return count

    def prune(self, *, now: float | None = None) -> list[MemoryRecord]:
        """Apply optional TTL/capacity rules and return removed records.

        这一步让 working memory 更像真实短期记忆：过旧或超容量的信息会
        自动被挤出，避免每次都依赖外部手动调用 forget。
        """

        current_time = now if now is not None else time()
        victims: list[MemoryRecord] = []

        if self.ttl_seconds is not None:
            victims.extend(
                record
                for record in self.records
                if (current_time - record.created_at) > self.ttl_seconds
            )

        kept = [record for record in self.records if record not in victims]
        if self.max_records is not None and len(kept) > self.max_records:
            ranked = sorted(
                kept,
                key=lambda item: (item.importance, item.access_count, item.created_at),
                reverse=True,
            )
            keep_ids = {record.memory_id for record in ranked[: self.max_records]}
            victims.extend(record for record in kept if record.memory_id not in keep_ids)

        if victims:
            victim_ids = {record.memory_id for record in victims}
            self.records = [
                record for record in self.records if record.memory_id not in victim_ids
            ]
            self._persist()
        return victims

    def _apply_record_changes(self, record: MemoryRecord, changes: dict[str, Any]) -> None:
        if "content" in changes and changes["content"] is not None:
            record.content = str(changes["content"]).strip()
        if "importance" in changes and changes["importance"] is not None:
            record.importance = float(changes["importance"])
        metadata = changes.get("metadata")
        if isinstance(metadata, dict):
            record.metadata.update(metadata)

    def _persist(self) -> None:
        if self.persistence is not None:
            self.persistence.save(self.records)

    def _tokenize(self, text: str) -> Counter[str]:
        """Tokenize Chinese-ish text without pulling in a heavy dependency.

        这里不是要替代 spaCy/embedding，而是给 working memory 一个可解释的
        smoke-test 检索能力：英文按词，中文保留连续片段和 2-gram。
        """

        normalized = text.lower()
        parts = re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]+", normalized)
        terms: list[str] = []
        for part in parts:
            terms.append(part)
            if re.fullmatch(r"[\u4e00-\u9fff]+", part):
                terms.extend(part[i : i + 2] for i in range(max(0, len(part) - 1)))
        return Counter(term for term in terms if term)

    def _overlap(
        self,
        query_terms: Counter[str],
        content_terms: Counter[str],
    ) -> float:
        """Raw keyword-overlap relevance, before importance/recency weighting."""

        if not query_terms or not content_terms:
            return 0.0

        overlap = 0.0
        for term, query_count in query_terms.items():
            if term in content_terms:
                overlap += min(query_count, content_terms[term])
        return overlap


class SemanticMemoryStore:
    """Vector-backed long-term semantic memory.

    默认情况下它会复用 hello-agents 已安装的 embedding 与 Qdrant 封装；
    测试时可以注入 fake embedder/vector_store，避免依赖外部 Qdrant 服务。
    """

    def __init__(
        self,
        *,
        embedder: Any | None = None,
        vector_store: Any | None = None,
        graph_store: Any | None = None,
        entity_extractor: Any | None = None,
        enable_graph_index: bool = True,
        collection_name: str = "my_semantic_memory",
        vector_size: int = 384,
        half_life_seconds: float = 0.0,
        recency_weight: float = 0.0,
        access_gain: float = 0.5,
        restore_existing: bool = True,
    ) -> None:
        self.embedder = embedder or self._build_default_embedder()
        self.vector_store = vector_store or self._build_default_vector_store(
            collection_name=collection_name,
            vector_size=vector_size,
        )
        self.graph_store = graph_store
        self.entity_extractor = (
            entity_extractor
            if entity_extractor is not None
            else SpacyEntityExtractor()
        )
        self.enable_graph_index = enable_graph_index
        # 语义记忆是长期、稳定的抽象知识，默认不做时间衰减，只叠加重要性因子。
        self.half_life_seconds = half_life_seconds
        self.recency_weight = recency_weight
        self.access_gain = access_gain
        # 进程内保留一份引用，方便 遗忘/整合 枚举候选 + 访问强化回查；
        # 向量库本身不擅长全量 scroll，也不适合频繁写访问计数。
        self.records: list[MemoryRecord] = []
        if restore_existing:
            self.records = self._load_existing_records()

    def add(self, record: MemoryRecord) -> MemoryRecord:
        entities = self._extract_entities(record)
        if entities:
            record.metadata["entities"] = entities
        vector = self._encode_one(record.content)
        metadata = {
            "memory_id": record.memory_id,
            "content": record.content,
            "memory_type": record.memory_type,
            "importance": record.importance,
            "created_at": record.created_at,
            "access_count": record.access_count,
            "last_accessed": record.last_accessed,
            **record.metadata,
        }
        ok = self.vector_store.add_vectors(
            vectors=[vector],
            metadata=[metadata],
            ids=[record.memory_id],
        )
        if ok is False:
            raise RuntimeError("向量写入失败")
        self._index_semantic_graph(record, entities)
        self._upsert_local_record(record)
        return record

    def search(self, query: str, limit: int = 5) -> list[MemorySearchResult]:
        query_vector = self._encode_one(query)
        hits = self.vector_store.search_similar(
            query_vector=query_vector,
            limit=limit,
        )

        now = time()
        by_id = {r.memory_id: r for r in self.records}
        results: list[MemorySearchResult] = []
        seen_contents: set[str] = set()
        for hit in hits:
            meta = hit.get("metadata", {}) or {}
            content = str(meta.get("content", ""))
            if not content:
                continue
            if content in seen_contents:
                continue
            seen_contents.add(content)
            created_at = float(meta.get("created_at", 0.0))
            importance = float(meta.get("importance", 0.5))
            memory_id = str(meta.get("memory_id", hit.get("id")))
            # 优先回查进程内的真实记录，拿到累积的 access_count；回查不到再从 meta 重建。
            record = by_id.get(memory_id)
            if record is None:
                record = MemoryRecord(
                    content=content,
                    memory_type=str(meta.get("memory_type", "semantic")),
                    importance=importance,
                    metadata={key: value for key, value in meta.items() if key not in {
                        "content",
                        "memory_type",
                        "importance",
                        "created_at",
                        "memory_id",
                    }},
                    memory_id=memory_id,
                    created_at=created_at,
                    access_count=int(meta.get("access_count", 0)),
                )
            score = combined_score(
                float(hit.get("score", 0.0)),
                importance,
                created_at=created_at,
                now=now,
                half_life_seconds=self.half_life_seconds,
                recency_weight=self.recency_weight,
                access_count=record.access_count,
                access_gain=self.access_gain,
            )
            results.append(
                MemorySearchResult(record=record, score=score, source="semantic")
            )
        for item in results:
            item.record.reinforce(now)
            self._persist_access(item.record)
        return results

    def summary(self, limit: int = 5) -> list[MemoryRecord]:
        return sorted(
            self.records,
            key=lambda item: (item.importance, item.created_at),
            reverse=True,
        )[:limit]

    def all_records(self) -> list[MemoryRecord]:
        return list(self.records)

    def delete(self, memory_id: str) -> bool:
        found = False
        for index, record in enumerate(self.records):
            if record.memory_id == memory_id:
                del self.records[index]
                found = True
                break
        # 尽量把删除同步到向量库；不同后端方法名不一，缺失时静默跳过。
        deleter = getattr(self.vector_store, "delete_memories", None) or getattr(
            self.vector_store, "delete_vectors", None
        ) or getattr(
            self.vector_store, "delete", None
        )
        if deleter is not None:
            try:
                deleter(ids=[memory_id])
            except TypeError:
                deleter([memory_id])
            except Exception:
                pass
        return found

    def update(self, memory_id: str, **changes: Any) -> MemoryRecord | None:
        for record in self.records:
            if record.memory_id != memory_id:
                continue
            self._apply_record_changes(record, changes)
            # 语义记忆内容/重要性改变后，需要重新写入向量库，让后续检索用新文本。
            self.add(record)
            return record
        return None

    def clear(self) -> int:
        memory_ids = [record.memory_id for record in self.records]
        count = len(memory_ids)
        for memory_id in memory_ids:
            self.delete(memory_id)
        return count

    def _apply_record_changes(self, record: MemoryRecord, changes: dict[str, Any]) -> None:
        if "content" in changes and changes["content"] is not None:
            record.content = str(changes["content"]).strip()
        if "importance" in changes and changes["importance"] is not None:
            record.importance = float(changes["importance"])
        metadata = changes.get("metadata")
        if isinstance(metadata, dict):
            record.metadata.update(metadata)

    def _extract_entities(self, record: MemoryRecord) -> list[dict[str, Any]]:
        if not self.enable_graph_index or self.entity_extractor is None:
            return []
        try:
            raw_entities = self.entity_extractor.extract(record.content)
        except Exception:
            return []

        entities: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for item in raw_entities or []:
            name = str(item.get("name", "")).strip()
            entity_type = str(item.get("type", "Entity")).strip() or "Entity"
            if not name:
                continue
            key = (name, entity_type)
            if key in seen:
                continue
            entities.append({"name": name, "type": entity_type})
            seen.add(key)
        return entities

    def _index_semantic_graph(
        self,
        record: MemoryRecord,
        entities: list[dict[str, Any]],
    ) -> None:
        if not self.enable_graph_index or self.graph_store is None or not entities:
            return

        memory_entity_id = f"semantic:{record.memory_id}"
        self.graph_store.add_entity(
            entity_id=memory_entity_id,
            name=record.content[:80],
            entity_type="SemanticMemory",
            properties={
                "memory_id": record.memory_id,
                "content": record.content,
                "importance": record.importance,
                "created_at_ts": record.created_at,
            },
        )
        for entity in entities:
            entity_id = normalize_entity_id(entity["name"], entity["type"])
            self.graph_store.add_entity(
                entity_id=entity_id,
                name=entity["name"],
                entity_type=entity["type"],
                properties={"source": "semantic_memory"},
            )
            self.graph_store.add_relationship(
                from_entity_id=memory_entity_id,
                to_entity_id=entity_id,
                relationship_type="MENTIONS",
                properties={"memory_id": record.memory_id},
            )

    def _upsert_local_record(self, record: MemoryRecord) -> None:
        for index, existing in enumerate(self.records):
            if existing.memory_id == record.memory_id:
                self.records[index] = record
                return
        self.records.append(record)

    def _load_existing_records(self) -> list[MemoryRecord]:
        """Best-effort restore from vector-store payloads when the backend supports it."""

        loader = getattr(self.vector_store, "scroll", None)
        if loader is None and hasattr(self.vector_store, "client"):
            client = getattr(self.vector_store, "client")

            def loader(limit: int = 10_000) -> list[dict[str, Any]]:
                response = client.scroll(
                    collection_name=self.vector_store.collection_name,
                    limit=limit,
                    with_payload=True,
                    with_vectors=False,
                )
                points = response[0] if isinstance(response, tuple) else response
                return [
                    {
                        "id": point.id,
                        "metadata": point.payload or {},
                    }
                    for point in points
                ]

        if loader is None:
            rows = getattr(self.vector_store, "rows", [])
        else:
            try:
                rows = loader(limit=10_000)
            except TypeError:
                rows = loader()
            except Exception:
                rows = []

        records: list[MemoryRecord] = []
        seen: set[str] = set()
        for row in rows or []:
            meta = row.get("metadata", row.get("payload", row)) or {}
            if meta.get("memory_type") != "semantic":
                continue
            record = self._record_from_metadata(meta, row.get("id"))
            if record.memory_id not in seen:
                records.append(record)
                seen.add(record.memory_id)
        return records

    def _record_from_metadata(self, meta: dict[str, Any], fallback_id: Any = None) -> MemoryRecord:
        data = {
            "content": meta.get("content", ""),
            "memory_type": meta.get("memory_type", "semantic"),
            "importance": meta.get("importance", 0.5),
            "metadata": {
                key: value
                for key, value in meta.items()
                if key
                not in {
                    "content",
                    "memory_type",
                    "importance",
                    "created_at",
                    "memory_id",
                    "access_count",
                    "last_accessed",
                }
            },
            "memory_id": meta.get("memory_id", fallback_id),
            "created_at": meta.get("created_at", 0.0),
            "access_count": meta.get("access_count", 0),
            "last_accessed": meta.get("last_accessed", 0.0),
        }
        return record_from_dict(data)

    def _persist_access(self, record: MemoryRecord) -> None:
        """Best-effort access-state writeback for Qdrant-like stores."""

        payload = {
            "access_count": record.access_count,
            "last_accessed": record.last_accessed,
        }
        updater = getattr(self.vector_store, "update_metadata", None)
        if updater is not None:
            try:
                updater(record.memory_id, payload)
                return
            except Exception:
                pass
        # Fake stores in tests expose rows; updating them keeps behavior observable.
        rows = getattr(self.vector_store, "rows", None)
        if isinstance(rows, list):
            for row in rows:
                meta = row.get("metadata", {})
                if meta.get("memory_id") == record.memory_id or row.get("id") == record.memory_id:
                    meta.update(payload)
        client = getattr(self.vector_store, "client", None)
        if client is not None:
            try:
                from qdrant_client.http import models

                client.set_payload(
                    collection_name=self.vector_store.collection_name,
                    payload=payload,
                    points=models.FilterSelector(
                        filter=models.Filter(
                            must=[
                                models.FieldCondition(
                                    key="memory_id",
                                    match=models.MatchValue(value=record.memory_id),
                                )
                            ]
                        )
                    ),
                    wait=True,
                )
            except Exception:
                pass

    def _encode_one(self, text: str) -> list[float]:
        encoded = self.embedder.encode([text])
        if hasattr(encoded, "tolist"):
            encoded = encoded.tolist()
        if encoded and hasattr(encoded[0], "tolist"):
            encoded = [item.tolist() for item in encoded]
        if encoded and isinstance(encoded[0], (int, float)):
            return [float(item) for item in encoded]
        return [float(item) for item in encoded[0]]

    def _build_default_embedder(self) -> Any:
        from hello_agents.memory.embedding import get_text_embedder

        return get_text_embedder()

    def _build_default_vector_store(self, *, collection_name: str, vector_size: int) -> Any:
        import os

        from hello_agents.memory.storage.qdrant_store import QdrantVectorStore

        return QdrantVectorStore(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY") or None,
            collection_name=collection_name,
            vector_size=vector_size,
            distance="cosine",
        )


class EpisodicMemoryStore:
    """Graph-backed memory for concrete events and experiences.

    情景记忆强调“发生过什么”。这里把每条事件保存为 Episode 节点，
    再把用户节点与事件节点用 EXPERIENCED 关系连起来。真实环境可接
    Neo4j；测试环境注入 fake graph_store。
    """

    def __init__(
        self,
        *,
        user_id: str = "default",
        graph_store: Any | None = None,
        half_life_seconds: float = 86400.0,
        recency_weight: float = 0.4,
        access_gain: float = 0.5,
        restore_existing: bool = True,
    ) -> None:
        self.user_id = user_id
        self.graph_store = graph_store or self._build_default_graph_store()
        self.records: list[MemoryRecord] = []
        # 情景记忆强调“最近发生的事更相关”，默认 1 天半衰、近因占 40%。
        self.half_life_seconds = half_life_seconds
        self.recency_weight = recency_weight
        self.access_gain = access_gain
        if restore_existing:
            self.records = self._load_existing_records()

    def add(self, record: MemoryRecord) -> MemoryRecord:
        user_entity_id = f"user:{self.user_id}"
        episode_entity_id = f"episode:{record.memory_id}"
        properties = {
            "memory_id": record.memory_id,
            "content": record.content,
            "memory_type": record.memory_type,
            "importance": record.importance,
            "created_at_ts": record.created_at,
            "access_count": record.access_count,
            "last_accessed": record.last_accessed,
            **record.metadata,
        }

        self.graph_store.add_entity(
            entity_id=user_entity_id,
            name=self.user_id,
            entity_type="User",
            properties={"user_id": self.user_id},
        )
        ok = self.graph_store.add_entity(
            entity_id=episode_entity_id,
            name=record.content,
            entity_type="Episode",
            properties=properties,
        )
        if ok is False:
            raise RuntimeError("情景记忆事件节点写入失败")

        self.graph_store.add_relationship(
            from_entity_id=user_entity_id,
            to_entity_id=episode_entity_id,
            relationship_type="EXPERIENCED",
            properties={
                "memory_type": record.memory_type,
                "importance": record.importance,
            },
        )
        self._upsert_local_record(record)
        return record

    def search(self, query: str, limit: int = 5) -> list[MemorySearchResult]:
        rows = self.graph_store.search_entities_by_name(
            name_pattern=query,
            entity_types=["Episode"],
            limit=limit,
        )

        now = time()
        by_id = {r.memory_id: r for r in self.records}
        results: list[MemorySearchResult] = []
        for row in rows:
            content = str(row.get("content") or row.get("name") or "")
            if not content:
                continue
            importance = float(row.get("importance", 0.5))
            created_at = float(row.get("created_at_ts", 0.0))
            memory_id = str(row.get("memory_id", row.get("id", "")))
            # 优先回查进程内真实记录，拿到累积的 access_count；回查不到再从图行重建。
            record = by_id.get(memory_id)
            if record is None:
                record = MemoryRecord(
                    content=content,
                    memory_type=str(row.get("memory_type", "episodic")),
                    importance=importance,
                    metadata={
                        key: value
                        for key, value in row.items()
                        if key not in {"content", "name", "memory_type", "importance", "memory_id"}
                    },
                    memory_id=memory_id,
                    created_at=created_at,
                    access_count=int(row.get("access_count", 0)),
                )
            # 图命中本身不带相关性强弱，基础分取 1.0，靠重要性 + 时间近因 + 访问强化区分。
            score = combined_score(
                1.0,
                importance,
                created_at=created_at,
                now=now,
                half_life_seconds=self.half_life_seconds,
                recency_weight=self.recency_weight,
                access_count=record.access_count,
                access_gain=self.access_gain,
            )
            results.append(
                MemorySearchResult(record=record, score=score, source="episodic")
            )

        if results:
            results.sort(key=lambda item: item.score, reverse=True)
            top = results[:limit]
            for item in top:
                item.record.reinforce(now)
                self._persist_access(item.record)
            return top

        # 如果图存储只支持精确/正则搜索失败，当前进程内记录还能提供兜底。
        # 兜底用同样的近因/强化配置，并直接喂入真实 records，让强化落在它们身上。
        fallback = WorkingMemoryStore(
            half_life_seconds=self.half_life_seconds,
            recency_weight=self.recency_weight,
            access_gain=self.access_gain,
        )
        fallback.records = self.records
        return [
            MemorySearchResult(record=item.record, score=item.score, source="episodic")
            for item in fallback.search(query, limit=limit)
        ]

    def summary(self, limit: int = 5) -> list[MemoryRecord]:
        return sorted(
            self.records,
            key=lambda item: (item.importance, item.created_at),
            reverse=True,
        )[:limit]

    def all_records(self) -> list[MemoryRecord]:
        return list(self.records)

    def delete(self, memory_id: str) -> bool:
        found = False
        for index, record in enumerate(self.records):
            if record.memory_id == memory_id:
                del self.records[index]
                found = True
                break
        # 尝试同步删除图节点；fake/真实图存储方法名不同，缺失时静默跳过。
        deleter = getattr(self.graph_store, "delete_entity", None)
        if deleter is not None:
            try:
                deleter(entity_id=f"episode:{memory_id}")
            except TypeError:
                deleter(f"episode:{memory_id}")
            except Exception:
                pass
        return found

    def update(self, memory_id: str, **changes: Any) -> MemoryRecord | None:
        for record in self.records:
            if record.memory_id != memory_id:
                continue
            self._apply_record_changes(record, changes)
            self.add(record)
            return record
        return None

    def clear(self) -> int:
        memory_ids = [record.memory_id for record in self.records]
        count = len(memory_ids)
        for memory_id in memory_ids:
            self.delete(memory_id)
        return count

    def _apply_record_changes(self, record: MemoryRecord, changes: dict[str, Any]) -> None:
        if "content" in changes and changes["content"] is not None:
            record.content = str(changes["content"]).strip()
        if "importance" in changes and changes["importance"] is not None:
            record.importance = float(changes["importance"])
        metadata = changes.get("metadata")
        if isinstance(metadata, dict):
            record.metadata.update(metadata)

    def _upsert_local_record(self, record: MemoryRecord) -> None:
        for index, existing in enumerate(self.records):
            if existing.memory_id == record.memory_id:
                self.records[index] = record
                return
        self.records.append(record)

    def _load_existing_records(self) -> list[MemoryRecord]:
        loader = getattr(self.graph_store, "search_entities_by_name", None)
        if loader is None:
            return []
        try:
            rows = loader(name_pattern=".*", entity_types=["Episode"], limit=10_000)
        except Exception:
            return []
        records: list[MemoryRecord] = []
        seen: set[str] = set()
        for row in rows:
            memory_id = str(row.get("memory_id", row.get("id", ""))).replace("episode:", "")
            if not memory_id or memory_id in seen:
                continue
            record = MemoryRecord(
                content=str(row.get("content") or row.get("name") or ""),
                memory_type=str(row.get("memory_type", "episodic")),
                importance=float(row.get("importance", 0.5)),
                metadata={
                    key: value
                    for key, value in row.items()
                    if key
                    not in {
                        "content",
                        "name",
                        "memory_type",
                        "importance",
                        "memory_id",
                        "created_at_ts",
                        "access_count",
                        "last_accessed",
                    }
                },
                memory_id=memory_id,
                created_at=float(row.get("created_at_ts", 0.0) or 0.0),
                access_count=int(row.get("access_count", 0) or 0),
                last_accessed=float(row.get("last_accessed", 0.0) or 0.0),
            )
            records.append(record)
            seen.add(memory_id)
        return records

    def _persist_access(self, record: MemoryRecord) -> None:
        payload = {
            "access_count": record.access_count,
            "last_accessed": record.last_accessed,
        }
        updater = getattr(self.graph_store, "update_entity", None)
        if updater is not None:
            try:
                updater(entity_id=f"episode:{record.memory_id}", properties=payload)
                return
            except Exception:
                pass
        entities = getattr(self.graph_store, "entities", None)
        if isinstance(entities, dict):
            entity = entities.get(f"episode:{record.memory_id}")
            if entity is not None:
                entity.update(payload)
        driver = getattr(self.graph_store, "driver", None)
        if driver is not None:
            try:
                with driver.session(database=self.graph_store.database) as session:
                    session.run(
                        """
                        MATCH (e:Entity {id: $entity_id})
                        SET e.access_count = $access_count,
                            e.last_accessed = $last_accessed
                        """,
                        entity_id=f"episode:{record.memory_id}",
                        **payload,
                    )
            except Exception:
                pass

    def _build_default_graph_store(self) -> Any:
        import os

        from hello_agents.memory.storage.neo4j_store import Neo4jGraphStore

        return Neo4jGraphStore(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            username=os.getenv("NEO4J_USERNAME", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "hello-agents-password"),
            database=os.getenv("NEO4J_DATABASE", "neo4j"),
        )


class PerceptualMemoryStore:
    """Store multimodal observations and their extracted text.

    感知记忆记录的是“用户给系统看到了什么”：图片、截图、音频、文件等。
    第一版先保存描述和元数据；搜索时同时查 content、file_path、
    modality 和 extracted_text。
    """

    def __init__(
        self,
        *,
        half_life_seconds: float = 86400.0,
        recency_weight: float = 0.2,
        access_gain: float = 0.5,
        persistence_path: str | None = None,
    ) -> None:
        self.persistence = (
            JsonMemoryPersistence(persistence_path) if persistence_path else None
        )
        self.records: list[MemoryRecord] = (
            self.persistence.load() if self.persistence is not None else []
        )
        # 复用工作记忆的分词与重叠打分；感知记忆默认 1 天半衰、近因占 20%。
        self._scorer = WorkingMemoryStore()
        self.half_life_seconds = half_life_seconds
        self.recency_weight = recency_weight
        self.access_gain = access_gain

    def add(self, record: MemoryRecord) -> MemoryRecord:
        self.records.append(record)
        self._persist()
        return record

    def search(self, query: str, limit: int = 5) -> list[MemorySearchResult]:
        query_terms = self._scorer._tokenize(query)
        now = time()
        results: list[MemorySearchResult] = []

        for record in self.records:
            searchable_text = " ".join(
                str(part)
                for part in [
                    record.content,
                    record.metadata.get("modality", ""),
                    record.metadata.get("file_path", ""),
                    record.metadata.get("extracted_text", ""),
                    record.metadata.get("description", ""),
                ]
                if part
            )
            content_terms = self._scorer._tokenize(searchable_text)
            overlap = self._scorer._overlap(query_terms, content_terms)
            if overlap <= 0:
                continue
            score = combined_score(
                overlap,
                record.importance,
                created_at=record.created_at,
                now=now,
                half_life_seconds=self.half_life_seconds,
                recency_weight=self.recency_weight,
                access_count=record.access_count,
                access_gain=self.access_gain,
            )
            results.append(
                MemorySearchResult(record=record, score=score, source="perceptual")
            )

        results.sort(key=lambda item: item.score, reverse=True)
        top = results[:limit]
        for item in top:
            item.record.reinforce(now)
        if top:
            self._persist()
        return top

    def summary(self, limit: int = 5) -> list[MemoryRecord]:
        return sorted(
            self.records,
            key=lambda item: (item.importance, item.created_at),
            reverse=True,
        )[:limit]

    def all_records(self) -> list[MemoryRecord]:
        return list(self.records)

    def delete(self, memory_id: str) -> bool:
        for index, record in enumerate(self.records):
            if record.memory_id == memory_id:
                del self.records[index]
                self._persist()
                return True
        return False

    def update(self, memory_id: str, **changes: Any) -> MemoryRecord | None:
        for record in self.records:
            if record.memory_id != memory_id:
                continue
            if "content" in changes and changes["content"] is not None:
                record.content = str(changes["content"]).strip()
            if "importance" in changes and changes["importance"] is not None:
                record.importance = float(changes["importance"])
            metadata = changes.get("metadata")
            if isinstance(metadata, dict):
                record.metadata.update(metadata)
            self._persist()
            return record
        return None

    def clear(self) -> int:
        count = len(self.records)
        self.records = []
        self._persist()
        return count

    def search_by_embedding(
        self,
        query_vector: list[float],
        *,
        modality: str | None = None,
        limit: int = 5,
    ) -> list[MemorySearchResult]:
        """Search perceptual records by injected multimodal embeddings."""

        now = time()
        results: list[MemorySearchResult] = []
        for record in self.records:
            if modality and record.metadata.get("modality") != modality:
                continue
            vector = record.metadata.get("embedding")
            if not isinstance(vector, list) or not vector:
                continue
            score = self._dot(query_vector, [float(item) for item in vector])
            if score <= 0:
                continue
            results.append(
                MemorySearchResult(
                    record=record,
                    score=combined_score(
                        score,
                        record.importance,
                        created_at=record.created_at,
                        now=now,
                        half_life_seconds=self.half_life_seconds,
                        recency_weight=self.recency_weight,
                        access_count=record.access_count,
                        access_gain=self.access_gain,
                    ),
                    source="perceptual_embedding",
                )
            )
        results.sort(key=lambda item: item.score, reverse=True)
        top = results[:limit]
        for item in top:
            item.record.reinforce(now)
        if top:
            self._persist()
        return top

    def _dot(self, left: list[float], right: list[float]) -> float:
        length = min(len(left), len(right))
        return sum(left[index] * right[index] for index in range(length))

    def _persist(self) -> None:
        if self.persistence is not None:
            self.persistence.save(self.records)
