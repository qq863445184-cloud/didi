from app.my_memory_system import (
    EpisodicMemoryStore,
    MemoryRecord,
    PerceptualMemoryStore,
    SemanticMemoryStore,
    WorkingMemoryStore,
)


class FakeEmbedder:
    def encode(self, texts):
        return [[1.0, 0.0] if "后端" in text else [0.0, 1.0] for text in texts]


class FakeVectorStore:
    def __init__(self) -> None:
        self.rows = []

    def add_vectors(self, vectors, metadata, ids=None):
        ids = ids or [str(index) for index in range(len(vectors))]
        for vector, meta, memory_id in zip(vectors, metadata, ids):
            self.rows.append({"id": memory_id, "vector": vector, "metadata": meta})
        return True

    def search_similar(self, query_vector, limit=5, score_threshold=None, where=None):
        scored = []
        for row in self.rows:
            score = sum(a * b for a, b in zip(query_vector, row["vector"]))
            if score > 0:
                scored.append({"id": row["id"], "score": score, "metadata": row["metadata"]})
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:limit]

    def delete_memories(self, memory_ids):
        self.rows = [
            row for row in self.rows
            if row["metadata"].get("memory_id") not in set(memory_ids)
        ]


class FakeGraphStore:
    def __init__(self) -> None:
        self.entities = {}
        self.relationships = []

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
            {"from": from_entity_id, "to": to_entity_id, "type": relationship_type}
        )
        return True

    def search_entities_by_name(self, name_pattern, entity_types=None, limit=20):
        types = set(entity_types or [])
        rows = []
        for entity in self.entities.values():
            if types and entity["type"] not in types:
                continue
            if name_pattern == ".*" or name_pattern in entity["name"]:
                rows.append(entity)
        return rows[:limit]


def test_working_memory_persists_records_and_access_state(tmp_path):
    path = tmp_path / "working.json"
    store = WorkingMemoryStore(persistence_path=str(path))
    store.add(MemoryRecord(content="用户是后端开发者", importance=0.9))
    store.search("后端开发者")

    restored = WorkingMemoryStore(persistence_path=str(path))

    assert len(restored.records) == 1
    assert restored.records[0].content == "用户是后端开发者"
    assert restored.records[0].access_count == 1


def test_perceptual_memory_persists_multimodal_metadata(tmp_path):
    path = tmp_path / "perceptual.json"
    store = PerceptualMemoryStore(persistence_path=str(path))
    store.add(
        MemoryRecord(
            content="用户上传了架构图",
            memory_type="perceptual",
            metadata={
                "modality": "image",
                "file_path": "diagram.png",
                "embedding": [0.1, 0.2],
            },
        )
    )

    restored = PerceptualMemoryStore(persistence_path=str(path))

    assert restored.records[0].metadata["modality"] == "image"
    assert restored.records[0].metadata["embedding"] == [0.1, 0.2]


def test_semantic_memory_restores_from_vector_metadata_and_writes_access_back():
    vector_store = FakeVectorStore()
    original = SemanticMemoryStore(
        embedder=FakeEmbedder(),
        vector_store=vector_store,
        restore_existing=False,
    )
    saved = original.add(
        MemoryRecord(content="用户是后端开发者", memory_type="semantic", importance=0.8)
    )

    restored = SemanticMemoryStore(embedder=FakeEmbedder(), vector_store=vector_store)
    restored.search("后端")

    assert restored.records[0].memory_id == saved.memory_id
    assert restored.records[0].access_count == 1
    assert vector_store.rows[0]["metadata"]["access_count"] == 1


def test_episodic_memory_restores_from_graph_and_writes_access_back():
    graph_store = FakeGraphStore()
    original = EpisodicMemoryStore(
        user_id="u",
        graph_store=graph_store,
        restore_existing=False,
    )
    saved = original.add(MemoryRecord(content="完成了记忆系统测试", memory_type="episodic"))

    restored = EpisodicMemoryStore(user_id="u", graph_store=graph_store)
    restored.search("记忆系统测试")

    assert restored.records[0].memory_id == saved.memory_id
    assert restored.records[0].access_count == 1
    assert graph_store.entities[f"episode:{saved.memory_id}"]["access_count"] == 1
