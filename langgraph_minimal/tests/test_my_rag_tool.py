from app.my_memory_system import MyRAGTool


class FakeEmbedder:
    def encode(self, texts):
        vectors = []
        for text in texts:
            if "LangGraph" in text or "图" in text or "审批" in text:
                vectors.append([1.0, 0.0, 0.0])
            elif "AutoGen" in text or "群聊" in text:
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])
        return vectors


class FakeVectorStore:
    def __init__(self) -> None:
        self.rows = []

    def add_vectors(self, vectors, metadata, ids=None):
        ids = ids or [str(index) for index in range(len(vectors))]
        for vector, meta, row_id in zip(vectors, metadata, ids):
            self.rows.append({"id": row_id, "vector": vector, "metadata": meta})
        return True

    def search_similar(self, query_vector, limit=5, score_threshold=None, where=None):
        scored = []
        for row in self.rows:
            if where:
                matched = all(row["metadata"].get(key) == value for key, value in where.items())
                if not matched:
                    continue
            score = sum(a * b for a, b in zip(query_vector, row["vector"]))
            if score_threshold is not None and score < score_threshold:
                continue
            scored.append({"id": row["id"], "score": score, "metadata": row["metadata"]})
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:limit]


class FakeLLM:
    def __init__(self) -> None:
        self.last_messages = []

    def invoke(self, messages, **kwargs):
        self.last_messages = messages
        return "LangGraph 适合金融审批，因为图结构让流程可追踪、可审计。"


def build_tool(llm=None):
    return MyRAGTool(
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
        llm=llm or FakeLLM(),
        collection_name="test_rag",
    )


def test_my_rag_tool_adds_text_chunks_and_searches_them():
    tool = build_tool()

    add_result = tool.run(
        {
            "action": "add_text",
            "text": "LangGraph 适合构建有状态、多步骤、可循环的 Agent 工作流。它适合金融审批和代码审查。",
            "document_id": "langgraph_intro",
            "namespace": "chapter8",
            "chunk_size": 30,
            "chunk_overlap": 5,
        }
    )
    search_result = tool.run(
        {
            "action": "search",
            "query": "金融审批 图 工作流",
            "namespace": "chapter8",
            "limit": 3,
        }
    )

    assert "已添加到知识库" in add_result
    assert "搜索结果" in search_result
    assert "langgraph_intro" in search_result
    assert "金融审批" in search_result
    assert tool.trace_events[0]["stage"] == "rag.add_text"
    assert tool.trace_events[1]["stage"] == "rag.search"


def test_my_rag_tool_ask_uses_retrieved_context_for_llm_answer():
    llm = FakeLLM()
    tool = build_tool(llm=llm)
    tool.run(
        {
            "action": "add_text",
            "text": "LangGraph 通过节点和边表达流程，因此适合金融审批这类需要审计的场景。",
            "document_id": "langgraph_audit",
            "namespace": "chapter8",
        }
    )

    answer = tool.run(
        {
            "action": "ask",
            "question": "LangGraph 为什么适合金融审批？",
            "namespace": "chapter8",
            "limit": 2,
        }
    )

    assert "智能问答结果" in answer
    assert "可追踪" in answer
    assert "参考来源" in answer
    assert "langgraph_audit" in answer
    assert "相关上下文" in llm.last_messages[-1]["content"]


def test_my_rag_tool_filters_by_namespace():
    tool = build_tool()
    tool.run(
        {
            "action": "add_text",
            "text": "LangGraph 适合金融审批。",
            "document_id": "doc_a",
            "namespace": "a",
        }
    )
    tool.run(
        {
            "action": "add_text",
            "text": "LangGraph 也可以用于代码审查。",
            "document_id": "doc_b",
            "namespace": "b",
        }
    )

    result = tool.run({"action": "search", "query": "LangGraph", "namespace": "b", "limit": 5})

    assert "doc_b" in result
    assert "doc_a" not in result


def test_my_rag_tool_adds_markdown_document(tmp_path):
    tool = build_tool()
    doc_path = tmp_path / "chapter8.md"
    doc_path.write_text(
        "# LangGraph\n\nLangGraph 使用图结构表达审批流程，适合金融风控系统。",
        encoding="utf-8",
    )

    add_result = tool.run(
        {
            "action": "add_document",
            "file_path": str(doc_path),
            "namespace": "chapter8",
            "chunk_size": 24,
            "chunk_overlap": 4,
        }
    )
    search_result = tool.run(
        {
            "action": "search",
            "query": "金融风控 图结构",
            "namespace": "chapter8",
        }
    )

    assert "文档已添加到知识库" in add_result
    assert "chapter8.md" in add_result
    assert "搜索结果" in search_result
    assert "金融风控" in search_result
    assert tool.trace_events[0]["stage"] == "rag.add_document"


def test_my_rag_tool_rejects_missing_document():
    tool = build_tool()

    result = tool.run({"action": "add_document", "file_path": "missing.md"})

    assert "错误" in result
    assert "文件不存在" in result
