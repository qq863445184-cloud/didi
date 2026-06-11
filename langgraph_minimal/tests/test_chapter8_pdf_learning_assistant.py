from app.my_memory_system import PDFLearningAssistant

from app.my_memory_system import MyMemoryManager, MyRAGTool, WorkingMemoryStore


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


def build_pdf_assistant():
    rag_tool = MyRAGTool(
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
        llm=FakeLLM(),
        collection_name="pdf_learning_assistant_test",
    )
    manager = MyMemoryManager(
        user_id="pdf_learner",
        stores={
            "working": WorkingMemoryStore(),
            "episodic": WorkingMemoryStore(),
            "semantic": WorkingMemoryStore(),
            "perceptual": WorkingMemoryStore(),
        },
    )
    return PDFLearningAssistant(
        rag_tool=rag_tool,
        memory_manager=manager,
        namespace="chapter8_pdf_learning",
        session_id="pdf-session",
    )


def test_pdf_learning_assistant_exposes_chapter8_method_names(tmp_path):
    assistant = build_pdf_assistant()
    pdf_like_path = tmp_path / "rag_notes.md"
    pdf_like_path.write_text(
        "RAG 学习流程包括文档入库、语义检索、基于上下文回答和引用追踪。",
        encoding="utf-8",
    )

    load_output = assistant.load_pdf(str(pdf_like_path))
    answer_output = assistant.ask_question("RAG 学习流程包括什么？")
    note_output = assistant.add_note("RAG 学习要关注引用来源。")
    recall_output = assistant.recall("引用来源")
    stats = assistant.get_stats()
    report = assistant.generate_report()

    assert "文档已添加到知识库" in load_output
    assert "文档入库" in answer_output
    assert "已保存学习笔记" in note_output
    assert "引用来源" in recall_output
    assert stats["learning_metrics"]["documents_loaded"] == 1
    assert report["title"] == "学习报告"
