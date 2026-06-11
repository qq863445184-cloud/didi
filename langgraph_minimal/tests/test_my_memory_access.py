from time import time

from app.my_memory_system import (
    EpisodicMemoryStore,
    MemoryRecord,
    MyMemoryManager,
    MyMemoryTool,
    PerceptualMemoryStore,
    SemanticMemoryStore,
    WorkingMemoryStore,
    access_factor,
    combined_score,
)


def test_access_factor_saturates_and_is_monotonic():
    # 0 次访问不加成（向后兼容）
    assert access_factor(0) == 1.0
    # 越访问越高，但单调饱和、有上限
    f1 = access_factor(1, gain=0.5)
    f5 = access_factor(5, gain=0.5)
    f100 = access_factor(100, gain=0.5)
    assert 1.0 < f1 < f5 < f100
    assert f100 < 1.0 + 0.5  # 永远不超过 1 + gain 的上限
    # gain=0 时关闭强化
    assert access_factor(10, gain=0.0) == 1.0


def test_combined_score_access_count_boosts_score():
    base = combined_score(1.0, 0.5, access_count=0)
    boosted = combined_score(1.0, 0.5, access_count=5)
    assert boosted > base


def test_memory_record_reinforce_increments_count_and_time():
    record = MemoryRecord(content="x")
    assert record.access_count == 0
    assert record.last_accessed == 0.0

    record.reinforce(now=123.0)
    assert record.access_count == 1
    assert record.last_accessed == 123.0

    record.reinforce(now=456.0)
    assert record.access_count == 2
    assert record.last_accessed == 456.0


def test_working_search_reinforces_returned_records():
    store = WorkingMemoryStore()
    store.add(MemoryRecord(content="用户在学习 Agent 记忆系统"))

    store.search("Agent 记忆系统")
    store.search("Agent 记忆系统")

    # 同一条记忆被检索两次，access_count 累积
    assert store.records[0].access_count == 2
    assert store.records[0].last_accessed > 0.0


def test_unmatched_records_are_not_reinforced():
    store = WorkingMemoryStore()
    store.add(MemoryRecord(content="后端接口设计"))
    store.add(MemoryRecord(content="完全无关的内容"))

    store.search("后端接口")

    matched = next(r for r in store.records if "后端" in r.content)
    unmatched = next(r for r in store.records if "无关" in r.content)
    assert matched.access_count == 1
    assert unmatched.access_count == 0  # 没命中就不强化


def test_repeated_access_lifts_ranking_over_equal_peer():
    # 关掉时间因子，单独观察访问强化对排名的影响
    store = WorkingMemoryStore(recency_weight=0.0, access_gain=0.5)
    now = time()
    a = MemoryRecord(content="后端接口设计要点", importance=0.5)
    b = MemoryRecord(content="后端接口设计要点", importance=0.5)
    a.created_at = b.created_at = now
    store.add(a)
    store.add(b)

    # 先单独“反复想起” a：手动强化它若干次
    for _ in range(5):
        a.reinforce(now)

    results = store.search("后端接口设计", limit=2)

    assert results[0].record is a  # 被频繁访问的 a 排在前面
    assert results[0].score > results[1].score


def test_memory_tool_search_reinforces_through_manager():
    manager = MyMemoryManager(stores={"working": WorkingMemoryStore()})
    tool = MyMemoryTool(manager=manager)
    tool.run({"action": "add", "content": "用户叫王小明，是后端开发者"})

    tool.run({"action": "search", "query": "后端开发者"})
    tool.run({"action": "search", "query": "后端开发者"})

    record = manager.stores["working"].records[0]
    assert record.access_count == 2


class FakeEmbedder:
    def encode(self, texts):
        # 含“后端/接口”给一个固定方向，便于命中；其余正交
        return [
            [1.0, 0.0] if ("后端" in t or "接口" in t) else [0.0, 1.0]
            for t in texts
        ]


class FakeVectorStore:
    def __init__(self) -> None:
        self.rows = []

    def add_vectors(self, vectors, metadata, ids=None):
        ids = ids or [str(i) for i in range(len(vectors))]
        for vec, meta, mid in zip(vectors, metadata, ids):
            self.rows.append({"id": mid, "vector": vec, "metadata": meta})
        return True

    def search_similar(self, query_vector, limit=5, score_threshold=None, where=None):
        scored = []
        for row in self.rows:
            score = sum(a * b for a, b in zip(query_vector, row["vector"]))
            if score <= 0:
                continue
            scored.append({"id": row["id"], "score": score, "metadata": row["metadata"]})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:limit]


def test_semantic_search_accumulates_access_on_persistent_record():
    store = SemanticMemoryStore(embedder=FakeEmbedder(), vector_store=FakeVectorStore())
    saved = store.add(MemoryRecord(content="用户是后端开发者，关注接口设计", memory_type="semantic"))

    # 关键：命中是从向量库 meta 重建的，但回查 self.records 让计数落在真实记录上
    first = store.search("后端接口")
    second = store.search("后端接口")

    assert first and first[0].record.memory_id == saved.memory_id
    # 跨两次检索，真实记录的 access_count 应累积为 2，而非每次重建归零
    assert saved.access_count == 2
    # 返回的命中就是同一个持久对象
    assert second[0].record is saved


def test_semantic_repeated_access_boosts_score():
    store = SemanticMemoryStore(embedder=FakeEmbedder(), vector_store=FakeVectorStore())
    store.add(MemoryRecord(content="后端接口设计", memory_type="semantic", importance=0.5))

    s1 = store.search("后端接口")[0].score
    s2 = store.search("后端接口")[0].score  # 第二次时 access_count 已 +1
    assert s2 > s1


class FakeGraphStore:
    def __init__(self) -> None:
        self.entities = {}
        self.relationships = []

    def add_entity(self, entity_id, name, entity_type, properties=None):
        self.entities[entity_id] = {"id": entity_id, "name": name, "type": entity_type, **(properties or {})}
        return True

    def add_relationship(self, from_entity_id, to_entity_id, relationship_type, properties=None):
        self.relationships.append({"from": from_entity_id, "to": to_entity_id, "type": relationship_type})
        return True

    def search_entities_by_name(self, name_pattern, entity_types=None, limit=20):
        types = set(entity_types or [])
        rows = [
            e for e in self.entities.values()
            if (not types or e["type"] in types) and name_pattern in e["name"]
        ]
        return rows[:limit]


def test_episodic_search_accumulates_access_on_persistent_record():
    store = EpisodicMemoryStore(user_id="u", graph_store=FakeGraphStore())
    saved = store.add(MemoryRecord(content="完成了语义记忆测试", memory_type="episodic"))

    store.search("语义记忆测试")
    hits = store.search("语义记忆测试")

    # 图命中也按 memory_id 回查真实记录，计数累积
    assert saved.access_count == 2
    assert hits[0].record is saved


def test_perceptual_search_reinforces_returned_records():
    store = PerceptualMemoryStore()
    saved = store.add(
        MemoryRecord(
            content="Python 代码截图",
            memory_type="perceptual",
            metadata={"modality": "image", "extracted_text": "def add(a, b): return a + b"},
        )
    )

    store.search("代码截图")
    store.search("代码截图")

    assert saved.access_count == 2
    assert saved.last_accessed > 0.0

