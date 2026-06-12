from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


from app.my_memory_system import EpisodicMemoryStore, MemoryRecord, SemanticMemoryStore


class SmokeEmbedder:
    def encode(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


class SmokeVectorStore:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def add_vectors(self, vectors, metadata, ids=None):
        ids = ids or [str(index) for index in range(len(vectors))]
        for vector, meta, row_id in zip(vectors, metadata, ids):
            self.rows.append({"id": row_id, "vector": vector, "metadata": meta})
        return True

    def search_similar(self, query_vector, limit=5, score_threshold=None, where=None):
        # 让 semantic smoke 明确验证图谱召回，而不是向量召回。
        return []


class SmokeEntityExtractor:
    def __init__(self, smoke_id: str) -> None:
        self.smoke_id = smoke_id

    def extract(self, text: str) -> list[dict[str, str]]:
        if self.smoke_id not in text:
            return []
        return [
            {"name": f"Chapter8Neo4jSmoke_{self.smoke_id}", "type": "SmokeEntity"},
        ]


class FakeGraphStore:
    def __init__(self) -> None:
        self.entities: dict[str, dict[str, Any]] = {}
        self.relationships: list[dict[str, Any]] = []

    def add_entity(self, entity_id, name, entity_type, properties=None):
        self.entities[entity_id] = {
            "id": entity_id,
            "name": name,
            "type": entity_type,
            **(properties or {}),
        }
        return True

    def add_relationship(self, from_entity_id, to_entity_id, relationship_type, properties=None):
        self.relationships.append(
            {
                "from": from_entity_id,
                "to": to_entity_id,
                "type": relationship_type,
                "properties": properties or {},
            }
        )
        return True

    def search_entities_by_name(self, name_pattern, entity_types=None, limit=20):
        allowed_types = set(entity_types or [])
        rows = []
        for entity in self.entities.values():
            if allowed_types and entity.get("type") not in allowed_types:
                continue
            haystack = " ".join(str(value) for value in entity.values())
            if name_pattern == ".*" or name_pattern in haystack:
                rows.append(entity)
        return rows[:limit]


def run_smoke(*, real: bool = False, smoke_id: str | None = None, cleanup: bool = True) -> dict[str, Any]:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    smoke_id = smoke_id or f"chapter8_smoke_{uuid.uuid4().hex[:10]}"
    graph_store = _build_graph_store(real=real)

    semantic_store = SemanticMemoryStore(
        embedder=SmokeEmbedder(),
        vector_store=SmokeVectorStore(),
        graph_store=graph_store,
        entity_extractor=SmokeEntityExtractor(smoke_id),
        restore_existing=False,
    )
    semantic_record = semantic_store.add(
        MemoryRecord(
            content=f"{smoke_id}: Python Agent memory writes extracted entities into Neo4j graph.",
            memory_type="semantic",
            importance=0.9,
        )
    )
    semantic_hits = semantic_store.search(f"Chapter8Neo4jSmoke_{smoke_id}", limit=3)

    episodic_store = EpisodicMemoryStore(
        user_id=f"user_{smoke_id}",
        graph_store=graph_store,
        restore_existing=False,
    )
    episodic_record = episodic_store.add(
        MemoryRecord(
            content=f"{smoke_id}: User finished one chapter 8 Neo4j episodic memory smoke.",
            memory_type="episodic",
            importance=0.8,
        )
    )
    episodic_hits = episodic_store.search(smoke_id, limit=3)

    cleanup_result = _cleanup_graph(graph_store, smoke_id) if cleanup and real else {"attempted": False}
    return {
        "mode": "real" if real else "fake",
        "smoke_id": smoke_id,
        "semantic": {
            "memory_id": semantic_record.memory_id,
            "entities": semantic_record.metadata.get("entities", []),
            "graph_hit_count": len(semantic_hits),
            "first_hit_source": semantic_hits[0].source if semantic_hits else "",
            "first_hit": semantic_hits[0].record.content if semantic_hits else "",
        },
        "episodic": {
            "memory_id": episodic_record.memory_id,
            "graph_hit_count": len(episodic_hits),
            "first_hit_source": episodic_hits[0].source if episodic_hits else "",
            "first_hit": episodic_hits[0].record.content if episodic_hits else "",
        },
        "cleanup": cleanup_result,
    }


def _build_graph_store(*, real: bool) -> Any:
    if not real:
        return FakeGraphStore()

    from hello_agents.memory.storage.neo4j_store import Neo4jGraphStore

    return Neo4jGraphStore(
        uri=os.environ["NEO4J_URI"],
        username=os.environ["NEO4J_USERNAME"],
        password=os.environ["NEO4J_PASSWORD"],
        database=os.getenv("NEO4J_DATABASE", "neo4j"),
    )


def _cleanup_graph(graph_store: Any, smoke_id: str) -> dict[str, Any]:
    driver = getattr(graph_store, "driver", None)
    database = getattr(graph_store, "database", os.getenv("NEO4J_DATABASE", "neo4j"))
    if driver is None:
        return {"attempted": False, "reason": "graph store has no driver"}
    try:
        with driver.session(database=database) as session:
            result = session.run(
                """
                MATCH (n:Entity)
                WHERE n.id CONTAINS $smoke_id
                   OR n.name CONTAINS $smoke_id
                   OR coalesce(n.content, '') CONTAINS $smoke_id
                   OR coalesce(n.user_id, '') CONTAINS $smoke_id
                DETACH DELETE n
                RETURN count(n) AS deleted
                """,
                smoke_id=smoke_id,
            )
            row = result.single()
        return {"attempted": True, "deleted": int(row["deleted"] if row else 0)}
    except Exception as exc:
        return {"attempted": True, "error": str(exc)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Chapter 8 Neo4j graph memory smoke")
    parser.add_argument("--real", action="store_true", help="connect to NEO4J_* from .env")
    parser.add_argument("--smoke-id", default="")
    parser.add_argument("--no-cleanup", action="store_true")
    args = parser.parse_args()

    result = run_smoke(
        real=args.real,
        smoke_id=args.smoke_id or None,
        cleanup=not args.no_cleanup,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
