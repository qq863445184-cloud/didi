import subprocess
import sys

from scripts.chapter8_document_learning_demo import build_demo, build_sample_document


def test_chapter8_document_learning_demo_runs_learning_workflow(tmp_path):
    assistant = build_demo()
    document_path = build_sample_document(tmp_path)

    load_result = assistant.load_document(document_path, chunk_size=48, chunk_overlap=8)
    answer = assistant.ask("RAG 学习流程包括什么？", limit=3)
    note_result = assistant.add_note("RAG 学习要同时关注知识库证据和学习记忆。")
    recall_result = assistant.recall("知识库证据 学习记忆")
    report = assistant.generate_report()

    assert load_result.document_id == "chapter8_learning_note.md"
    assert "文档入库" in answer.answer
    assert answer.references
    assert answer.retrieved_chunks
    assert "RAG 学习流程" in answer.retrieved_chunks[0]["content"]
    document = assistant.rag_tool.document_store.get_document(
        "chapter8_learning_note.md",
        namespace=assistant.namespace,
    )
    chunks = assistant.rag_tool.document_store.list_chunks(
        "chapter8_learning_note.md",
        namespace=assistant.namespace,
    )
    assert document["parser"] == "plain_text"
    assert chunks
    assert any("记忆闭环" in chunk["content"] for chunk in chunks)
    assert "已保存学习笔记" in note_result
    assert "知识库证据" in recall_result
    assert report["title"] == "学习报告"
    assert report["learning_metrics"]["documents_loaded"] == 1
    assert any(event["stage"] == "learning.load_document" for event in assistant.trace_events)
    assert any(event["stage"] == "learning.ask" for event in assistant.trace_events)
    assert any(event["stage"] == "learning.add_note" for event in assistant.trace_events)


def test_chapter8_document_learning_demo_script_exits_cleanly():
    result = subprocess.run(
        [sys.executable, "scripts/chapter8_document_learning_demo.py"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert "[3] retrieved chunks" in result.stdout
    assert "[4] document store" in result.stdout
