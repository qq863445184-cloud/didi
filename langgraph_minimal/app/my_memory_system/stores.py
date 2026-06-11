from __future__ import annotations

import re
from collections import Counter
from time import time
from typing import Any, Protocol

from .models import MemoryRecord, MemorySearchResult
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
    ) -> None:
        self.records: list[MemoryRecord] = []
        # 工作记忆偏短期，默认 1 小时半衰、近因占基础分 30%。
        self.half_life_seconds = half_life_seconds
        self.recency_weight = recency_weight
        # 访问强化上限增益：被反复命中的记忆最多获得 (1 + access_gain) 倍加成。
        self.access_gain = access_gain

    def add(self, record: MemoryRecord) -> MemoryRecord:
        self.records.append(record)
        return record

    def search(self, query: str, limit: int = 5) -> list[MemorySearchResult]:
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
        return top

    def summary(self, limit: int = 5) -> list[MemoryRecord]:
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
                return True
        return False

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
        collection_name: str = "my_semantic_memory",
        vector_size: int = 384,
        half_life_seconds: float = 0.0,
        recency_weight: float = 0.0,
        access_gain: float = 0.5,
    ) -> None:
        self.embedder = embedder or self._build_default_embedder()
        self.vector_store = vector_store or self._build_default_vector_store(
            collection_name=collection_name,
            vector_size=vector_size,
        )
        # 语义记忆是长期、稳定的抽象知识，默认不做时间衰减，只叠加重要性因子。
        self.half_life_seconds = half_life_seconds
        self.recency_weight = recency_weight
        self.access_gain = access_gain
        # 进程内保留一份引用，方便 遗忘/整合 枚举候选 + 访问强化回查；
        # 向量库本身不擅长全量 scroll，也不适合频繁写访问计数。
        self.records: list[MemoryRecord] = []

    def add(self, record: MemoryRecord) -> MemoryRecord:
        vector = self._encode_one(record.content)
        metadata = {
            "memory_id": record.memory_id,
            "content": record.content,
            "memory_type": record.memory_type,
            "importance": record.importance,
            "created_at": record.created_at,
            **record.metadata,
        }
        ok = self.vector_store.add_vectors(
            vectors=[vector],
            metadata=[metadata],
            ids=[record.memory_id],
        )
        if ok is False:
            raise RuntimeError("向量写入失败")
        self.records.append(record)
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
        return results

    def summary(self, limit: int = 5) -> list[MemoryRecord]:
        # Qdrant 更适合按 query 检索；第一版不做全量 scroll，保持接口明确。
        return []

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
        deleter = getattr(self.vector_store, "delete_vectors", None) or getattr(
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
    ) -> None:
        self.user_id = user_id
        self.graph_store = graph_store or self._build_default_graph_store()
        self.records: list[MemoryRecord] = []
        # 情景记忆强调“最近发生的事更相关”，默认 1 天半衰、近因占 40%。
        self.half_life_seconds = half_life_seconds
        self.recency_weight = recency_weight
        self.access_gain = access_gain

    def add(self, record: MemoryRecord) -> MemoryRecord:
        user_entity_id = f"user:{self.user_id}"
        episode_entity_id = f"episode:{record.memory_id}"
        properties = {
            "memory_id": record.memory_id,
            "content": record.content,
            "memory_type": record.memory_type,
            "importance": record.importance,
            "created_at_ts": record.created_at,
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
        self.records.append(record)
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
    ) -> None:
        self.records: list[MemoryRecord] = []
        # 复用工作记忆的分词与重叠打分；感知记忆默认 1 天半衰、近因占 20%。
        self._scorer = WorkingMemoryStore()
        self.half_life_seconds = half_life_seconds
        self.recency_weight = recency_weight
        self.access_gain = access_gain

    def add(self, record: MemoryRecord) -> MemoryRecord:
        self.records.append(record)
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
                return True
        return False
