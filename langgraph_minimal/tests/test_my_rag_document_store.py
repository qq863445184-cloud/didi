from app.my_memory_system import MyRAGTool, SQLiteDocumentStore


class FakeEmbedder:
    def encode(self, texts):
        return [[1.0, 0.0] for _ in texts]


class FakeVectorStore:
    def __init__(self) -> None:
        self.rows = []

    def add_vectors(self, vectors, metadata, ids=None):
        row_ids = ids or [str(index) for index in range(len(vectors))]
        for vector, meta, row_id in zip(vectors, metadata, row_ids):
            self.rows.append({"id": row_id, "vector": vector, "metadata": meta})
        return True

    def search_similar(self, query_vector, limit=5, score_threshold=None, where=None):
        results = []
        for row in self.rows:
            if where and not all(row["metadata"].get(key) == value for key, value in where.items()):
                continue
            results.append({"id": row["id"], "score": 1.0, "metadata": row["metadata"]})
        return results[:limit]

    def delete_vectors(self, *, where):
        before = len(self.rows)
        self.rows = [
            row
            for row in self.rows
            if not all(row["metadata"].get(key) == value for key, value in where.items())
        ]
        return before - len(self.rows)


class FakeLLM:
    def invoke(self, messages, **kwargs):
        return "answer"


def test_sqlite_document_store_saves_documents_and_chunks(tmp_path):
    store = SQLiteDocumentStore(tmp_path / "rag_docs.sqlite3")

    store.upsert_document(
        document_id="chapter8.md",
        namespace="chapter8",
        source_path="D:/docs/chapter8.md",
        parser="plain_text",
        metadata={"title": "第八章"},
    )
    store.replace_chunks(
        document_id="chapter8.md",
        namespace="chapter8",
        chunks=[
            {
                "chunk_id": "chunk-0",
                "chunk_index": 0,
                "content": "# 第八章：记忆与检索",
                "section_title": "第八章：记忆与检索",
                "start_char": 0,
                "end_char": 10,
            },
            {
                "chunk_id": "chunk-1",
                "chunk_index": 1,
                "content": "## RAG 问答\n\nRAG 问答包括文档导入和向量检索。",
                "section_title": "RAG 问答",
                "start_char": 11,
                "end_char": 40,
            },
        ],
    )

    document = store.get_document("chapter8.md", namespace="chapter8")
    chunks = store.list_chunks("chapter8.md", namespace="chapter8")

    assert document["document_id"] == "chapter8.md"
    assert document["parser"] == "plain_text"
    assert document["metadata"]["title"] == "第八章"
    assert len(chunks) == 2
    assert chunks[1]["section_title"] == "RAG 问答"
    assert "向量检索" in chunks[1]["content"]


def test_my_rag_tool_writes_document_metadata_to_sqlite_store(tmp_path):
    document_store = SQLiteDocumentStore(tmp_path / "rag_docs.sqlite3")
    tool = MyRAGTool(
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
        llm=FakeLLM(),
        document_store=document_store,
        collection_name="test_rag_docs",
    )
    doc_path = tmp_path / "chapter8.md"
    doc_path.write_text(
        "# 第八章：记忆与检索\n\n## RAG 问答\n\nRAG 问答包括文档导入、切块入库和向量检索。",
        encoding="utf-8",
    )

    result = tool.run(
        {
            "action": "add_document",
            "file_path": str(doc_path),
            "namespace": "chapter8",
            "chunk_size": 64,
            "chunk_overlap": 8,
        }
    )

    document = document_store.get_document("chapter8.md", namespace="chapter8")
    chunks = document_store.list_chunks("chapter8.md", namespace="chapter8")

    assert "文档已添加到知识库" in result
    assert document["source_path"] == str(doc_path)
    assert document["parser"] == "plain_text"
    assert chunks
    assert chunks[0]["document_id"] == "chapter8.md"
    assert any(chunk["section_title"] == "RAG 问答" for chunk in chunks)


def test_sqlite_document_store_filters_by_namespace(tmp_path):
    store = SQLiteDocumentStore(tmp_path / "rag_docs.sqlite3")
    store.upsert_document(document_id="same.md", namespace="a", parser="plain_text")
    store.upsert_document(document_id="same.md", namespace="b", parser="plain_text")
    store.replace_chunks(
        document_id="same.md",
        namespace="b",
        chunks=[
            {
                "chunk_id": "b-0",
                "chunk_index": 0,
                "content": "命名空间 b 的内容",
            }
        ],
    )

    assert store.get_document("same.md", namespace="a")["namespace"] == "a"
    assert store.get_document("same.md", namespace="b")["namespace"] == "b"
    assert store.list_chunks("same.md", namespace="a") == []
    assert store.list_chunks("same.md", namespace="b")[0]["content"] == "命名空间 b 的内容"


def test_my_rag_tool_deletes_document_from_metadata_and_vectors(tmp_path):
    document_store = SQLiteDocumentStore(tmp_path / "rag_docs.sqlite3")
    vector_store = FakeVectorStore()
    tool = MyRAGTool(
        embedder=FakeEmbedder(),
        vector_store=vector_store,
        llm=FakeLLM(),
        document_store=document_store,
        collection_name="test_rag_docs",
    )
    tool.run(
        {
            "action": "add_text",
            "text": "RAG 删除文档时要同步清理 chunk 元数据和向量索引。",
            "document_id": "delete_me.md",
            "namespace": "chapter8",
            "chunk_size": 20,
            "chunk_overlap": 0,
        }
    )

    result = tool.run(
        {
            "action": "delete_document",
            "document_id": "delete_me.md",
            "namespace": "chapter8",
        }
    )

    assert "文档已删除" in result
    assert document_store.get_document("delete_me.md", namespace="chapter8") is None
    assert document_store.list_chunks("delete_me.md", namespace="chapter8") == []
    assert not any(row["metadata"]["document_id"] == "delete_me.md" for row in vector_store.rows)
    assert "delete_me.md" not in tool.run({"action": "search", "query": "RAG 删除文档", "namespace": "chapter8"})
