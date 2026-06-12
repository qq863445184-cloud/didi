from app.my_memory_system import (
    DocumentLearningAssistant,
    MyMemoryManager,
    MyRAGTool,
    PerceptualMemoryStore,
    WorkingMemoryStore,
)


class FakeEmbedder:
    def encode(self, texts):
        vectors = []
        for text in texts:
            if "RAG" in text or "检索" in text or "知识库" in text:
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.0, 1.0])
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
        hits = []
        for row in self.rows:
            if where and not all(row["metadata"].get(key) == value for key, value in where.items()):
                continue
            score = sum(a * b for a, b in zip(query_vector, row["vector"]))
            if score_threshold is not None and score < score_threshold:
                continue
            hits.append({"id": row["id"], "score": score, "metadata": row["metadata"]})
        hits.sort(key=lambda item: item["score"], reverse=True)
        return hits[:limit]


class FakeLLM:
    def invoke(self, messages, **kwargs):
        return "RAG 学习流程包括文档入库、语义检索、基于上下文回答和引用追踪。"


def build_assistant():
    rag_tool = MyRAGTool(
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
        llm=FakeLLM(),
        collection_name="learning_assistant_test",
    )
    manager = MyMemoryManager(
        user_id="learner",
        stores={
            "working": WorkingMemoryStore(),
            "episodic": WorkingMemoryStore(),
            "semantic": WorkingMemoryStore(),
            "perceptual": PerceptualMemoryStore(),
        },
    )
    return DocumentLearningAssistant(
        rag_tool=rag_tool,
        memory_manager=manager,
        namespace="chapter8_learning",
        session_id="session-1",
    )


def test_document_learning_assistant_ingests_document_and_records_learning_event(tmp_path):
    assistant = build_assistant()
    doc_path = tmp_path / "rag_notes.md"
    doc_path.write_text(
        "第八章 RAG 学习流程：文档入库、语义检索、上下文回答、引用追踪。",
        encoding="utf-8",
    )

    result = assistant.load_document(doc_path, chunk_size=30, chunk_overlap=5)

    assert "文档已添加到知识库" in result.raw_output
    assert result.document_id == "rag_notes.md"
    assert any(event["stage"] == "rag.add_document" for event in result.trace)

    episodic_records = assistant.memory_manager.summary(memory_type="episodic")
    perceptual_records = assistant.memory_manager.summary(memory_type="perceptual")
    semantic_records = assistant.memory_manager.summary(memory_type="semantic")
    assert any("加载学习文档" in record.content for record in episodic_records)
    assert episodic_records[0].metadata["session_id"] == "session-1"
    assert any("Document loaded for learning" in record.content for record in perceptual_records)
    assert any("Document summary for learning" in record.content for record in semantic_records)


def test_document_learning_assistant_asks_with_rag_and_memory_trace(tmp_path):
    assistant = build_assistant()
    doc_path = tmp_path / "rag_notes.md"
    doc_path.write_text(
        "RAG 学习流程包括文档入库、语义检索、基于上下文回答和引用追踪。",
        encoding="utf-8",
    )
    assistant.load_document(doc_path, chunk_size=40, chunk_overlap=5)

    result = assistant.ask("RAG 学习流程包括什么？")

    assert "文档入库" in result.answer
    assert result.references
    assert result.retrieved_chunks
    assert "RAG 学习流程" in result.retrieved_chunks[0]["content"]
    assert any(event["stage"] == "rag.ask" for event in result.trace)
    assert any(event["stage"] == "manager.search" for event in result.trace)
    assert any(event["stage"] == "learning.ask" for event in result.trace)

    working_records = assistant.memory_manager.summary(memory_type="working")
    episodic_records = assistant.memory_manager.summary(memory_type="episodic")
    assert any("当前学习问题" in record.content for record in working_records)
    assert any("完成一次文档问答" in record.content for record in episodic_records)


def test_document_learning_assistant_builds_review_report_from_memory():
    assistant = build_assistant()
    assistant.memory_manager.add(
        content="加载学习文档：rag_notes.md",
        memory_type="episodic",
        importance=0.7,
        metadata={"session_id": "session-1"},
    )
    assistant.memory_manager.add(
        content="当前学习问题：RAG 学习流程包括什么？",
        memory_type="working",
        importance=0.6,
        metadata={"session_id": "session-1"},
    )

    report = assistant.learning_report()

    assert "学习报告" in report
    assert "rag_notes.md" in report
    assert "RAG 学习流程" in report


def test_document_learning_assistant_supports_notes_recall_stats_and_report():
    assistant = build_assistant()

    note_result = assistant.add_note(
        "RAG 的关键是先召回相关片段，再让模型基于上下文回答。",
        importance=0.85,
    )
    recall_result = assistant.recall("相关片段 上下文")
    stats = assistant.get_stats()
    report = assistant.generate_report()

    assert "已保存学习笔记" in note_result
    assert "RAG 的关键" in recall_result
    assert stats["by_type"]["semantic"] == 1
    assert stats["by_type"]["working"] == 1
    assert stats["learning_metrics"]["concepts_learned"] == 1
    assert report["title"] == "学习报告"
    assert any("RAG 的关键" in item for item in report["memory_summary"]["semantic"])
    assert any(event["stage"] == "learning.add_note" for event in assistant.trace_events)
    assert any(event["stage"] == "learning.recall" for event in assistant.trace_events)


def test_document_learning_assistant_tracks_session_stats_and_saves_json_report(tmp_path):
    assistant = build_assistant()
    doc_path = tmp_path / "rag_notes.md"
    doc_path.write_text(
        "RAG 学习流程包括文档入库、语义检索、基于上下文回答和引用追踪。",
        encoding="utf-8",
    )

    assistant.load_document(doc_path, chunk_size=40, chunk_overlap=5)
    assistant.ask("RAG 学习流程包括什么？")
    assistant.add_note("RAG 学习要关注引用来源。")
    report_path = tmp_path / "learning_report.json"

    stats = assistant.get_stats()
    report = assistant.generate_report(save_to_file=True, file_path=report_path)

    assert stats["learning_metrics"]["documents_loaded"] == 1
    assert stats["learning_metrics"]["questions_asked"] == 1
    assert stats["learning_metrics"]["concepts_learned"] == 1
    assert stats["learning_metrics"]["current_document"] == "rag_notes.md"
    assert report["learning_metrics"] == stats["learning_metrics"]
    assert report["report_file"] == str(report_path)
    assert report_path.exists()
    assert "rag_notes.md" in report_path.read_text(encoding="utf-8")


def test_document_learning_assistant_routes_rag_memory_and_hybrid_questions(tmp_path):
    assistant = build_assistant()
    doc_path = tmp_path / "rag_notes.md"
    doc_path.write_text(
        "RAG 学习流程包括文档入库、语义检索、基于上下文回答和引用追踪。",
        encoding="utf-8",
    )
    assistant.load_document(doc_path, chunk_size=40, chunk_overlap=5)
    assistant.add_note("RAG 学习要关注知识库证据。")

    assert assistant.route_query("根据文档，RAG 学习流程包括什么？") == "rag"
    assert assistant.route_query("我之前保存了哪些学习笔记？") == "memory"
    assert assistant.route_query("结合文档和我的学习笔记说明 RAG。") == "hybrid"

    memory_answer = assistant.ask_auto("我之前保存了哪些学习笔记？")
    hybrid_answer = assistant.ask_auto("结合文档和我的学习笔记说明 RAG。")

    assert memory_answer.route == "memory"
    assert "RAG 学习要关注知识库证据" in memory_answer.answer
    assert hybrid_answer.route == "hybrid"
    assert hybrid_answer.references
    assert any(event["stage"] == "learning.route" for event in assistant.trace_events)


def test_document_learning_assistant_report_analyzes_trajectory_gaps_and_recommendations(tmp_path):
    assistant = build_assistant()
    doc_path = tmp_path / "rag_notes.md"
    doc_path.write_text(
        "RAG 学习流程包括文档入库、语义检索、基于上下文回答和引用追踪。",
        encoding="utf-8",
    )
    assistant.load_document(doc_path, chunk_size=40, chunk_overlap=5)
    assistant.ask_auto("根据文档，RAG 学习流程包括什么？")
    assistant.ask_auto("我之前保存了哪些学习笔记？")
    assistant.add_note("RAG 学习要关注引用来源。")

    report = assistant.generate_report()

    assert "analysis" in report
    assert "加载学习文档" in "\n".join(report["analysis"]["learning_trajectory"])
    assert any("Memory" in item or "记忆" in item for item in report["analysis"]["knowledge_gaps"])
    assert any("继续" in item or "建议" in item for item in report["analysis"]["recommendations"])
