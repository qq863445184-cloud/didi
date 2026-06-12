from app.my_memory_system import MyRAGTool, RAGQADemo


class FakeEmbedder:
    def encode(self, texts):
        vectors = []
        for text in texts:
            if "记忆" in text or "检索" in text or "RAG" in text:
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
        rows = []
        for row in self.rows:
            if where and not all(row["metadata"].get(key) == value for key, value in where.items()):
                continue
            score = sum(a * b for a, b in zip(query_vector, row["vector"]))
            if score_threshold is not None and score < score_threshold:
                continue
            rows.append({"id": row["id"], "score": score, "metadata": row["metadata"]})
        rows.sort(key=lambda item: item["score"], reverse=True)
        return rows[:limit]


class FakeLLM:
    def invoke(self, messages, **kwargs):
        prompt = messages[-1]["content"]
        if "改写成 3 个" in prompt:
            return "记忆系统\nRAG 检索\n知识库问答"
        return "RAG 问答流程包括文档导入、向量检索、上下文回答和引用展示。"


def test_rag_qa_demo_runs_end_to_end_with_references_and_trace(tmp_path):
    rag_tool = MyRAGTool(
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
        llm=FakeLLM(),
        collection_name="demo_test",
    )
    demo = RAGQADemo(rag_tool=rag_tool, namespace="chapter8_demo")
    doc_path = tmp_path / "chapter8.md"
    doc_path.write_text(
        "第八章介绍记忆与检索。RAG 问答流程包括文档导入、向量检索、上下文回答和引用展示。",
        encoding="utf-8",
    )

    ingest_result = demo.ingest_document(doc_path, chunk_size=40, chunk_overlap=5)
    result = demo.ask("RAG 问答流程包括什么？", enable_mqe=True)

    assert "文档已添加到知识库" in ingest_result
    assert "文档导入" in result.answer
    assert result.references
    assert "chapter8.md#chunk-" in result.references[0]
    assert result.retrieved_chunks
    assert result.retrieved_chunks[0]["document_id"] == "chapter8.md"
    assert "RAG 问答流程" in result.retrieved_chunks[0]["content"]
    assert any(event["stage"] == "rag.add_document" for event in result.trace)
    assert any(event["stage"] == "rag.expand_mqe" for event in result.trace)
    assert any(event["stage"] == "rag.ask" for event in result.trace)
