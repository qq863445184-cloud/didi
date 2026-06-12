from pathlib import Path

from app.my_memory_system import (
    DocumentLearningAssistant,
    DocumentLearningUI,
    DocumentParserPipeline,
    EpisodicMemoryStore,
    MyMemoryManager,
    MyMemoryTool,
    MyPerceptionTool,
    MyRAGTool,
    PDFLearningAssistant,
    PerceptualMemoryStore,
    SemanticMemoryStore,
    SQLiteDocumentStore,
    SpacyEntityExtractor,
    WorkingMemoryStore,
    build_gradio_app,
)


def test_chapter8_implementation_note_documents_operational_commands():
    doc_path = Path("docs/chapter8_memory_system.md")
    text = doc_path.read_text(encoding="utf-8")

    for keyword in [
        "working / semantic / episodic / perceptual",
        "MyRAGTool",
        "chapter8_backend_health.py",
        "chapter8_multimodal_worker.py",
        "chapter8_neo4j_graph_smoke.py",
        "Qdrant",
        "Neo4j",
    ]:
        assert keyword in text


def parameter_names(tool):
    return {parameter.name for parameter in tool.get_parameters()}


def test_chapter8_memory_tool_exposes_lifecycle_operations():
    tool = MyMemoryTool(manager=MyMemoryManager(stores={"working": WorkingMemoryStore()}))
    names = parameter_names(tool)

    assert tool.name == "memory"
    assert {"action", "content", "query", "memory_type", "memory_types"}.issubset(names)
    minimal_parameters = {
        "add": {"action": "add", "content": "记住这条学习记录"},
        "search": {"action": "search", "query": "学习记录"},
        "summary": {"action": "summary"},
        "stats": {"action": "stats"},
        "update": {"action": "update", "memory_id": "memory-id", "content": "更新"},
        "remove": {"action": "remove", "memory_id": "memory-id"},
        "clear_all": {"action": "clear_all"},
        "forget": {"action": "forget"},
        "consolidate": {"action": "consolidate"},
    }
    for parameters in minimal_parameters.values():
        assert tool.validate_parameters(parameters)


def test_chapter8_memory_layers_are_available():
    assert WorkingMemoryStore is not None
    assert SemanticMemoryStore is not None
    assert EpisodicMemoryStore is not None
    assert PerceptualMemoryStore is not None
    assert SpacyEntityExtractor is not None
    assert MyPerceptionTool is not None
    assert SQLiteDocumentStore is not None
    assert DocumentParserPipeline is not None


def test_chapter8_rag_tool_exposes_document_retrieval_actions():
    tool = MyRAGTool(embedder=object(), vector_store=object(), llm=object())
    names = parameter_names(tool)

    assert tool.name == "rag"
    assert {"action", "text", "file_path", "document_id", "namespace", "enable_mqe", "enable_hyde"}.issubset(names)
    assert tool.validate_parameters({"action": "stats"})
    assert tool.validate_parameters({"action": "add_text", "text": "hello"})
    assert tool.validate_parameters({"action": "add_document", "file_path": "a.md"})
    assert tool.validate_parameters({"action": "delete_document", "document_id": "a.md"})
    assert tool.validate_parameters({"action": "search", "query": "RAG"})
    assert tool.validate_parameters({"action": "ask", "question": "RAG?"})


def test_chapter8_learning_assistant_public_surface_matches_tutorial_case():
    for method_name in [
        "load_document",
        "ask",
        "route_query",
        "ask_auto",
        "add_note",
        "recall",
        "get_stats",
        "generate_report",
    ]:
        assert hasattr(DocumentLearningAssistant, method_name)

    for method_name in ["load_pdf", "ask_question", "add_note", "recall", "get_stats", "generate_report"]:
        assert hasattr(PDFLearningAssistant, method_name)


def test_chapter8_optional_web_ui_surface_is_present():
    for method_name in ["load_document", "ask", "add_note", "recall", "generate_report", "trace"]:
        assert hasattr(DocumentLearningUI, method_name)
    assert callable(build_gradio_app)
