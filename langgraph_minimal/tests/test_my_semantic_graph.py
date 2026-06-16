from app.my_memory_system import MemoryRecord, SemanticMemoryStore


class FakeEmbedder:
    def encode(self, texts):
        return [[1.0, 0.0] for _ in texts]


class FakeVectorStore:
    def __init__(self) -> None:
        self.rows = []

    def add_vectors(self, vectors, metadata, ids=None):
        ids = ids or [str(index) for index in range(len(vectors))]
        for vector, meta, row_id in zip(vectors, metadata, ids):
            self.rows.append({"id": row_id, "vector": vector, "metadata": meta})
        return True

    def search_similar(self, query_vector, limit=5, score_threshold=None, where=None):
        return []


class FakeEntityExtractor:
    def extract(self, text):
        assert "Python" in text
        return [
            {"name": "Python", "type": "Technology"},
            {"name": "Agent", "type": "Concept"},
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
            {
                "from": from_entity_id,
                "to": to_entity_id,
                "type": relationship_type,
                "properties": properties or {},
            }
        )
        return True

    def delete_entity(self, entity_id):
        if entity_id not in self.entities:
            return False
        del self.entities[entity_id]
        self.relationships = [
            rel
            for rel in self.relationships
            if rel["from"] != entity_id and rel["to"] != entity_id
        ]
        return True


def test_semantic_memory_writes_extracted_entities_to_graph():
    graph_store = FakeGraphStore()
    store = SemanticMemoryStore(
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
        graph_store=graph_store,
        entity_extractor=FakeEntityExtractor(),
    )

    record = store.add(
        MemoryRecord(
            content="Python 是构建 Agent 工具链时常用的语言。",
            memory_type="semantic",
        )
    )

    assert f"semantic:{record.memory_id}" in graph_store.entities
    assert "entity:Technology:Python" in graph_store.entities
    assert "entity:Concept:Agent" in graph_store.entities
    assert {
        ("semantic:" + record.memory_id, "entity:Technology:Python", "MENTIONS"),
        ("semantic:" + record.memory_id, "entity:Concept:Agent", "MENTIONS"),
    }.issubset(
        {(rel["from"], rel["to"], rel["type"]) for rel in graph_store.relationships}
    )
    assert record.metadata["entities"][0]["name"] == "Python"


def test_semantic_memory_can_disable_graph_extraction():
    graph_store = FakeGraphStore()
    store = SemanticMemoryStore(
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
        graph_store=graph_store,
        entity_extractor=None,
        enable_graph_index=False,
    )

    store.add(MemoryRecord(content="Python 与 Agent", memory_type="semantic"))

    assert graph_store.entities == {}
    assert graph_store.relationships == []


def test_semantic_memory_recalls_records_through_entity_graph_when_vector_misses():
    graph_store = FakeGraphStore()
    store = SemanticMemoryStore(
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
        graph_store=graph_store,
        entity_extractor=FakeEntityExtractor(),
    )
    store.add(
        MemoryRecord(
            content="Python 是构建 Agent 工具链时常用的语言。",
            memory_type="semantic",
            importance=0.9,
        )
    )

    results = store.search("Python", limit=3)

    assert len(results) == 1
    assert results[0].source == "semantic_graph"
    assert results[0].record.content == "Python 是构建 Agent 工具链时常用的语言。"
    assert results[0].record.access_count == 1


def test_semantic_memory_delete_removes_graph_node_and_relationships():
    graph_store = FakeGraphStore()
    store = SemanticMemoryStore(
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
        graph_store=graph_store,
        entity_extractor=FakeEntityExtractor(),
    )
    record = store.add(
        MemoryRecord(
            content="Python 是构建 Agent 工具链时常用的语言。",
            memory_type="semantic",
        )
    )

    deleted = store.delete(record.memory_id)

    assert deleted is True
    assert f"semantic:{record.memory_id}" not in graph_store.entities
    assert all(
        rel["from"] != f"semantic:{record.memory_id}"
        and rel["to"] != f"semantic:{record.memory_id}"
        for rel in graph_store.relationships
    )
