from app.my_memory_agent import MyMemoryAgent
from app.my_memory_system import (
    EpisodicMemoryStore,
    MyMemoryManager,
    MyMemoryTool,
    PerceptualMemoryStore,
    SemanticMemoryStore,
)


class FakeLLM:
    provider = "fake"

    def __init__(self) -> None:
        self.calls = []

    def invoke(self, messages, **kwargs):
        self.calls.append(messages)
        content = messages[-1]["content"]
        if "工具执行结果" in content and "已保存记忆" in content:
            return "记好了。"
        if "工具执行结果" in content and "王小明" in content:
            return "你叫王小明，是后端开发者。"
        if "请记住" in content:
            return (
                '<tool_call>{"name":"memory","parameters":'
                '{"action":"add","content":"用户叫王小明，是后端开发者","memory_type":"working","importance":0.9}}'
                "</tool_call>"
            )
        if "我叫什么" in content:
            return (
                '<tool_call>{"name":"memory","parameters":'
                '{"action":"search","query":"王小明 后端开发者","memory_type":"working","limit":3}}'
                "</tool_call>"
            )
        return "普通回答。"


def test_my_memory_tool_adds_and_searches_working_memory():
    tool = MyMemoryTool(user_id="test_user")

    add_result = tool.run(
        {
            "action": "add",
            "content": "用户叫王小明，是后端开发者，正在学习 Agent 记忆系统。",
            "memory_type": "working",
            "importance": 0.9,
        }
    )
    search_result = tool.run(
        {
            "action": "search",
            "query": "后端开发者",
            "memory_type": "working",
            "limit": 3,
        }
    )

    assert "已保存记忆" in add_result
    assert "找到 1 条相关记忆" in search_result
    assert "王小明" in search_result
    assert "后端开发者" in search_result
    assert tool.trace_events[0]["stage"] == "manager.add"
    assert tool.trace_events[1]["stage"] == "manager.search"


def test_my_memory_tool_summary_reports_memory_count():
    tool = MyMemoryTool(user_id="test_user")
    tool.run({"action": "add", "content": "用户喜欢简洁的 trace。"})

    result = tool.run({"action": "summary", "limit": 5})

    assert "记忆总数: 1" in result
    assert "用户喜欢简洁的 trace" in result


def test_my_memory_tool_exposes_hello_agents_tool_schema():
    tool = MyMemoryTool()

    schema = tool.to_dict()

    assert schema["name"] == "memory"
    assert any(item["name"] == "action" and item["required"] for item in schema["parameters"])
    assert tool.validate_parameters({"action": "search", "query": "后端开发者"})
    assert not tool.validate_parameters({})


def test_my_memory_tool_can_be_driven_by_my_memory_agent():
    memory_tool = MyMemoryTool(user_id="agent_user")
    agent = MyMemoryAgent(name="记忆助手", llm=FakeLLM(), memory_tool=memory_tool)

    add_response = agent.run("请记住我叫王小明，是后端开发者")
    search_response = agent.run("我叫什么？")

    assert add_response == "记好了。"
    assert search_response == "你叫王小明，是后端开发者。"
    assert len(memory_tool.manager.stores["working"].records) == 1
    assert agent.trace_events[-1]["stage"] == "memory_tool"


class FakeEmbedder:
    def encode(self, texts):
        vectors = []
        for text in texts:
            if "后端" in text or "接口" in text:
                vectors.append([1.0, 0.0, 0.0])
            elif "前端" in text:
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])
        return vectors


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
            if score_threshold is not None and score < score_threshold:
                continue
            scored.append(
                {
                    "id": row["id"],
                    "score": score,
                    "metadata": row["metadata"],
                }
            )
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:limit]


def test_my_memory_tool_routes_semantic_memory_to_vector_store():
    semantic_store = SemanticMemoryStore(
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
    )
    manager = MyMemoryManager(
        user_id="semantic_user",
        stores={"semantic": semantic_store},
    )
    tool = MyMemoryTool(manager=manager)

    tool.run(
        {
            "action": "add",
            "content": "用户是后端开发者，关注接口设计和 Agent 记忆系统。",
            "memory_type": "semantic",
            "importance": 0.9,
        }
    )
    tool.run(
        {
            "action": "add",
            "content": "用户也了解一些前端页面开发。",
            "memory_type": "semantic",
            "importance": 0.5,
        }
    )
    result = tool.run(
        {
            "action": "search",
            "query": "后端接口",
            "memory_type": "semantic",
            "limit": 1,
        }
    )

    assert "找到 1 条相关记忆" in result
    assert "后端开发者" in result
    assert "前端页面" not in result
    assert manager.trace_events[-1]["stage"] == "manager.search"


def test_semantic_memory_search_deduplicates_same_content_hits():
    semantic_store = SemanticMemoryStore(
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
    )
    manager = MyMemoryManager(
        user_id="semantic_user",
        stores={"semantic": semantic_store},
    )
    tool = MyMemoryTool(manager=manager)

    for _ in range(2):
        tool.run(
            {
                "action": "add",
                "content": "用户是后端开发者，关注接口设计。",
                "memory_type": "semantic",
            }
        )

    result = tool.run(
        {
            "action": "search",
            "query": "后端接口",
            "memory_type": "semantic",
            "limit": 3,
        }
    )

    assert "找到 1 条相关记忆" in result


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

    def search_entities_by_name(self, name_pattern, entity_types=None, limit=20):
        entity_types = set(entity_types or [])
        rows = []
        for entity in self.entities.values():
            if entity_types and entity["type"] not in entity_types:
                continue
            if name_pattern in entity["name"]:
                rows.append(entity)
        return rows[:limit]


def test_my_memory_tool_routes_episodic_memory_to_graph_store():
    graph_store = FakeGraphStore()
    episodic_store = EpisodicMemoryStore(user_id="episode_user", graph_store=graph_store)
    manager = MyMemoryManager(
        user_id="episode_user",
        stores={"episodic": episodic_store},
    )
    tool = MyMemoryTool(manager=manager)

    add_result = tool.run(
        {
            "action": "add",
            "content": "2026年6月10日，用户完成了自定义记忆系统的语义记忆测试。",
            "memory_type": "episodic",
            "importance": 0.8,
            "event_type": "milestone",
            "location": "本地开发环境",
        }
    )
    search_result = tool.run(
        {
            "action": "search",
            "query": "语义记忆测试",
            "memory_type": "episodic",
            "limit": 3,
        }
    )

    assert "已保存记忆" in add_result
    assert "找到 1 条相关记忆" in search_result
    assert "语义记忆测试" in search_result
    assert any(rel["type"] == "EXPERIENCED" for rel in graph_store.relationships)


def test_my_memory_tool_routes_perceptual_memory_with_file_metadata():
    perceptual_store = PerceptualMemoryStore()
    manager = MyMemoryManager(
        user_id="perceptual_user",
        stores={"perceptual": perceptual_store},
    )
    tool = MyMemoryTool(manager=manager)

    add_result = tool.run(
        {
            "action": "add",
            "content": "用户上传了一张 Python 代码截图。",
            "memory_type": "perceptual",
            "importance": 0.7,
            "modality": "image",
            "file_path": "./uploads/code_screenshot.png",
            "extracted_text": "def add(a, b): return a + b",
        }
    )
    search_result = tool.run(
        {
            "action": "search",
            "query": "函数定义 add",
            "memory_type": "perceptual",
            "limit": 3,
        }
    )

    assert "已保存记忆" in add_result
    assert "找到 1 条相关记忆" in search_result
    assert "Python 代码截图" in search_result
    assert "code_screenshot.png" in search_result
    assert manager.trace_events[-1]["stage"] == "manager.search"
