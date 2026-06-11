from scripts.chapter8_my_rag_qa_demo import build_demo, build_sample_document


def test_chapter8_my_rag_qa_demo_outputs_answer_references_and_trace(tmp_path):
    demo = build_demo()
    document_path = build_sample_document(tmp_path)

    ingest_result = demo.ingest_document(document_path, chunk_size=48, chunk_overlap=8)
    result = demo.ask("RAG 问答流程包括什么？", enable_mqe=True, limit=3)

    assert "文档已添加到知识库" in ingest_result
    assert "文档导入" in result.answer
    assert result.references
    assert any("chapter8_rag_note.md#chunk-" in reference for reference in result.references)
    assert [event["stage"] for event in result.trace] == [
        "rag.add_document",
        "rag.expand_mqe",
        "rag.merge_candidates",
        "rag.ask",
    ]
