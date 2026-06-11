from app.my_memory_system import (
    MyMemoryManager,
    MyMemoryTool,
    SemanticMemoryStore,
    WorkingMemoryStore,
)


class FakeEmbedder:
    def encode(self, texts):
        return [[float(len(text)), 0.0, 0.0] for text in texts]


class FakeVectorStore:
    def __init__(self) -> None:
        self.rows = []
        self.deleted_ids = []

    def add_vectors(self, vectors, metadata, ids=None):
        ids = ids or [str(index) for index in range(len(vectors))]
        for vector, meta, memory_id in zip(vectors, metadata, ids):
            self.rows.append({"id": memory_id, "vector": vector, "metadata": meta})
        return True

    def search_similar(self, query_vector, limit=5, score_threshold=None, where=None):
        return []

    def delete_vectors(self, ids):
        self.deleted_ids.extend(ids)
        self.rows = [row for row in self.rows if row["id"] not in set(ids)]
        return True


def _working_tool():
    manager = MyMemoryManager(stores={"working": WorkingMemoryStore()})
    return MyMemoryTool(manager=manager)


def test_forget_by_importance_drops_low_value_memories():
    tool = _working_tool()
    tool.run({"action": "add", "content": "重要：用户是后端负责人", "importance": 0.9})
    tool.run({"action": "add", "content": "闲聊：今天天气不错", "importance": 0.1})

    result = tool.run(
        {"action": "forget", "strategy": "importance", "importance_threshold": 0.3}
    )

    assert "已遗忘 1 条记忆" in result
    assert "今天天气不错" in result
    remaining = tool.manager.stores["working"].records
    assert len(remaining) == 1
    assert remaining[0].content == "重要：用户是后端负责人"
    assert tool.trace_events[-1]["stage"] == "manager.forget"


def test_forget_by_age_drops_stale_memories():
    store = WorkingMemoryStore()
    manager = MyMemoryManager(stores={"working": store})

    fresh = manager.add(content="刚刚发生的事", importance=0.5)
    old = manager.add(content="很久以前的事", importance=0.5)
    old.created_at = 1000.0  # 人为设为过期

    forgotten = manager.forget(
        strategy="age", max_age_seconds=3600, now=fresh.created_at + 10
    )

    assert [r.content for r in forgotten] == ["很久以前的事"]
    assert [r.content for r in store.records] == ["刚刚发生的事"]


def test_forget_by_capacity_keeps_top_n():
    tool = _working_tool()
    tool.run({"action": "add", "content": "记忆A", "importance": 0.9})
    tool.run({"action": "add", "content": "记忆B", "importance": 0.5})
    tool.run({"action": "add", "content": "记忆C", "importance": 0.2})

    result = tool.run({"action": "forget", "strategy": "capacity", "capacity": 2})

    assert "已遗忘 1 条记忆" in result
    remaining = {r.content for r in tool.manager.stores["working"].records}
    assert remaining == {"记忆A", "记忆B"}


def test_forget_age_requires_max_age():
    manager = MyMemoryManager(stores={"working": WorkingMemoryStore()})
    try:
        manager.forget(strategy="age")
    except ValueError as exc:
        assert "max_age_seconds" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("应当因缺少 max_age_seconds 报错")


def test_consolidate_promotes_high_importance_working_to_semantic():
    semantic_store = SemanticMemoryStore(
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
    )
    manager = MyMemoryManager(
        stores={
            "working": WorkingMemoryStore(),
            "semantic": semantic_store,
        }
    )
    tool = MyMemoryTool(manager=manager)

    tool.run({"action": "add", "content": "用户是后端负责人，长期合作", "importance": 0.9})
    tool.run({"action": "add", "content": "随口一提的小事", "importance": 0.2})

    result = tool.run(
        {
            "action": "consolidate",
            "memory_type": "working",
            "target_type": "semantic",
            "importance_threshold": 0.7,
        }
    )

    assert "已整合 1 条记忆到 semantic" in result
    # 高价值记忆已迁出工作记忆
    assert len(manager.stores["working"].records) == 1
    assert manager.stores["working"].records[0].content == "随口一提的小事"
    # 语义记忆收到整合后的副本，并带来源标记
    assert len(semantic_store.records) == 1
    promoted = semantic_store.records[0]
    assert promoted.memory_type == "semantic"
    assert promoted.metadata["consolidated_from"] == "working"
    assert tool.trace_events[-1]["stage"] == "manager.consolidate"


def test_consolidate_can_keep_source_records():
    manager = MyMemoryManager(
        stores={
            "working": WorkingMemoryStore(),
            "episodic": WorkingMemoryStore(),  # 用 working 当轻量目标，验证 delete_source=False
        }
    )
    manager.add(content="值得长期记住的结论", importance=0.95)

    consolidated = manager.consolidate(
        source_type="working",
        target_type="episodic",
        delete_source=False,
    )

    assert len(consolidated) == 1
    assert len(manager.stores["working"].records) == 1  # 源未删除
    assert len(manager.stores["episodic"].records) == 1
