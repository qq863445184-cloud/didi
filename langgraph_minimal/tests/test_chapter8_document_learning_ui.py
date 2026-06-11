import pytest

from scripts.chapter8_document_learning_demo import build_demo, build_sample_document
from app.my_memory_system.document_learning_ui import DocumentLearningUI, build_gradio_app


def test_document_learning_ui_callbacks_drive_assistant(tmp_path):
    ui = DocumentLearningUI(build_demo())
    document_path = build_sample_document(tmp_path)

    load_output = ui.load_document(str(document_path))
    answer_output = ui.ask("RAG 学习流程包括什么？")
    note_output = ui.add_note("RAG 学习要关注知识库证据。")
    recall_output = ui.recall("知识库证据")
    report_output = ui.generate_report()

    assert "文档已添加到知识库" in load_output
    assert "文档入库" in answer_output
    assert "已保存学习笔记" in note_output
    assert "知识库证据" in recall_output
    assert '"title": "学习报告"' in report_output


def test_build_gradio_app_reports_missing_optional_dependency():
    with pytest.raises(RuntimeError, match="pip install gradio"):
        build_gradio_app(build_demo())
