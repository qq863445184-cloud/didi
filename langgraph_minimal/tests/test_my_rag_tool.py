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


class AdvancedFakeEmbedder:
    def encode(self, texts):
        vectors = []
        for text in texts:
            if "风险控制" in text or "风控" in text or "贷款" in text:
                vectors.append([1.0, 0.0])
            elif "审批" in text:
                vectors.append([0.0, 1.0])
            else:
                vectors.append([0.0, 0.0])
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


class AdvancedFakeLLM:
    def __init__(self) -> None:
        self.calls = []

    def invoke(self, messages, **kwargs):
        self.calls.append(messages)
        prompt = messages[-1]["content"]
        if "改写成 3 个" in prompt:
            return "风险控制\n贷款风控\n信贷风险"
        if "生成一段可能出现在知识库中的假设性答案" in prompt:
            return "贷款审批系统通常包含风险控制、额度评估和合规检查。"
        return "基于增强检索得到答案。"


class FlatFakeEmbedder:
    def encode(self, texts):
        return [[1.0] for _ in texts]


def build_tool(llm=None):
    return MyRAGTool(
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
        llm=llm or FakeLLM(),
        collection_name="test_rag",
    )


def build_advanced_tool(llm=None):
    return MyRAGTool(
        embedder=AdvancedFakeEmbedder(),
        vector_store=FakeVectorStore(),
        llm=llm or AdvancedFakeLLM(),
        collection_name="test_rag_advanced",
    )


def build_flat_tool(llm=None):
    return MyRAGTool(
        embedder=FlatFakeEmbedder(),
        vector_store=FakeVectorStore(),
        llm=llm or FakeLLM(),
        collection_name="test_rag_flat",
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


def test_my_rag_tool_keeps_section_metadata_when_chunking_markdown():
    tool = build_tool()

    tool.run(
        {
            "action": "add_text",
            "text": "# 记忆系统\n\n工作记忆保存当前任务上下文。\n\n## RAG 检索\n\nRAG 检索负责从知识库召回相关片段。",
            "document_id": "memory_doc",
            "namespace": "chapter8",
            "chunk_size": 32,
            "chunk_overlap": 4,
        }
    )

    rows = tool.vector_store.rows
    assert len(rows) >= 2
    assert any(row["metadata"]["section_title"] == "记忆系统" for row in rows)
    assert any(row["metadata"]["section_title"] == "RAG 检索" for row in rows)
    assert all(row["metadata"]["start_char"] is not None for row in rows)
    assert all(row["metadata"]["end_char"] is not None for row in rows)


def test_my_rag_tool_keeps_markdown_section_together_without_fragment_overlap():
    tool = build_tool()

    tool.run(
        {
            "action": "add_text",
            "text": "\n\n".join(
                [
                    "# 第八章：记忆与检索",
                    "工作记忆保存当前任务上下文，语义记忆保存稳定知识，情景记忆保存具体事件。",
                    "## RAG 问答",
                    "RAG 问答通常包括文档导入、切块入库、向量检索、基于上下文生成答案和展示引用。",
                ]
            ),
            "document_id": "chapter8_note",
            "namespace": "chapter8",
            "chunk_size": 64,
            "chunk_overlap": 8,
        }
    )

    contents = [row["metadata"]["content"] for row in tool.vector_store.rows]

    assert any(
        content.startswith("## RAG 问答")
        and "文档导入、切块入库、向量检索" in content
        for content in contents
    )
    assert all(not content.startswith("八章：") for content in contents)
    assert all(not content.startswith("忆保存") for content in contents)


def test_my_rag_tool_splits_long_semantic_unit_with_overlap():
    tool = build_tool()
    long_text = "LangGraph" + "审批流程" * 20

    tool.run(
        {
            "action": "add_text",
            "text": long_text,
            "document_id": "long_doc",
            "namespace": "chapter8",
            "chunk_size": 24,
            "chunk_overlap": 6,
        }
    )

    rows = tool.vector_store.rows
    assert len(rows) > 1
    assert rows[1]["metadata"]["start_char"] < rows[0]["metadata"]["end_char"]


def test_my_rag_tool_rejects_missing_document():
    tool = build_tool()

    result = tool.run({"action": "add_document", "file_path": "missing.md"})

    assert "错误" in result
    assert "文件不存在" in result


def test_my_rag_tool_mqe_expands_query_and_merges_candidates():
    tool = build_advanced_tool()
    tool.run(
        {
            "action": "add_text",
            "text": "贷款审批系统需要风险控制和合规检查。",
            "document_id": "risk_doc",
            "namespace": "advanced",
        }
    )

    result = tool.run(
        {
            "action": "search",
            "query": "审批系统",
            "namespace": "advanced",
            "enable_mqe": True,
            "limit": 3,
        }
    )

    assert "搜索结果" in result
    assert "risk_doc" in result
    assert any(event["stage"] == "rag.expand_mqe" for event in tool.trace_events)
    assert tool.trace_events[-1]["candidate_queries"] >= 2


def test_my_rag_tool_hyde_retrieves_from_hypothetical_answer():
    tool = build_advanced_tool()
    tool.run(
        {
            "action": "add_text",
            "text": "贷款审批系统需要风险控制和合规检查。",
            "document_id": "hyde_doc",
            "namespace": "advanced",
        }
    )

    result = tool.run(
        {
            "action": "search",
            "query": "审批系统有哪些模块",
            "namespace": "advanced",
            "enable_hyde": True,
            "limit": 3,
        }
    )

    assert "搜索结果" in result
    assert "hyde_doc" in result
    assert any(event["stage"] == "rag.expand_hyde" for event in tool.trace_events)


def test_my_rag_tool_keyword_rerank_promotes_exact_business_terms():
    tool = build_flat_tool()
    tool.run(
        {
            "action": "add_text",
            "text": "通用退款流程要求客服先确认客户身份，再创建售后工单。",
            "document_id": "refund_policy",
            "namespace": "business",
        }
    )
    tool.run(
        {
            "action": "add_text",
            "text": "财务记录：发票 INV-2026-001 的金额为 1280 CNY，需要关联订单 ORD-2026-778。",
            "document_id": "invoice_record",
            "namespace": "business",
        }
    )

    result = tool.run(
        {
            "action": "search",
            "query": "发票 INV-2026-001 金额",
            "namespace": "business",
            "limit": 2,
            "enable_keyword_rerank": True,
        }
    )

    assert "invoice_record" in result
    assert tool.last_retrieved_chunks[0]["document_id"] == "invoice_record"
    assert tool.last_retrieved_chunks[0]["keyword_score"] > 0
    assert any(event["stage"] == "rag.keyword_rerank" for event in tool.trace_events)


def test_my_rag_tool_score_threshold_filters_low_scores():
    tool = build_advanced_tool()
    tool.run(
        {
            "action": "add_text",
            "text": "贷款审批系统需要风险控制。",
            "document_id": "risk_doc",
            "namespace": "advanced",
        }
    )

    result = tool.run(
        {
            "action": "search",
            "query": "无关查询",
            "namespace": "advanced",
            "score_threshold": 0.5,
        }
    )

    assert "未找到" in result
