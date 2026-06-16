import json

from app.my_memory_system import MemoryRAGDashboard, MyMemoryManager, MyPerceptionTool, PerceptualMemoryStore
from scripts.chapter8_memory_rag_dashboard_demo import (
    build_dashboard_demo,
    build_persistent_dashboard_demo,
)
from scripts.chapter8_memory_rag_dashboard_http_demo import (
    DashboardHTTPHandler,
    build_http_dashboard,
)


def test_build_dashboard_demo_wires_business_multimodal_stack():
    dashboard = build_dashboard_demo()

    assert dashboard.rag_tool is not None
    assert dashboard.perception_tool is not None
    assert dashboard.memory_manager is not None
    assert dashboard.rag_namespace == "business_multimodal"
    assert dashboard.backend_status()["backend_mode"] == "in-memory"
    assert "in-memory" in dashboard.backend_inventory()


def test_build_dashboard_demo_can_use_real_multimodal_extractors(tmp_path):
    calls = {"ocr": [], "asr": []}

    def fake_ocr(path):
        calls["ocr"].append(path.name)
        return "图片真实 OCR 文本：合同编号 IMG-REAL-001"

    def fake_asr(path):
        calls["asr"].append(path.name)
        return "音频真实 ASR 文本：会议提到 AUDIO-REAL-002"

    dashboard = build_dashboard_demo(
        prefer_real_multimodal=True,
        image_ocr=fake_ocr,
        audio_asr=fake_asr,
    )
    image = tmp_path / "contract.png"
    audio = tmp_path / "meeting.wav"
    image.write_bytes(b"fake image")
    audio.write_bytes(b"fake audio")

    image_result = dashboard.ingest_file(str(image), description="真实图片上传")
    audio_result = dashboard.ingest_file(str(audio), description="真实音频上传")

    assert calls == {"ocr": ["contract.png"], "asr": ["meeting.wav"]}
    assert "IMG-REAL-001" in image_result
    assert "AUDIO-REAL-002" in audio_result
    assert "extractor: image_ocr" in image_result
    assert "extractor: audio_asr" in audio_result

    answer = dashboard.ask("合同编号和会议编号分别是什么？")

    assert "IMG-REAL-001" in answer
    assert "AUDIO-REAL-002" in answer


def test_build_persistent_dashboard_demo_wires_document_store_and_memory_paths(tmp_path):
    dashboard = build_persistent_dashboard_demo(data_dir=tmp_path, strict_backends=False)

    assert dashboard.rag_tool.document_store.path == tmp_path / "rag_documents.sqlite3"
    assert dashboard.rag_tool.collection_name == "chapter8_persistent_rag"
    assert dashboard.memory_manager.stores["working"].persistence.path == tmp_path / "working_memory.json"
    assert dashboard.memory_manager.stores["perceptual"].persistence.path == tmp_path / "perceptual_memory.json"
    assert set(dashboard.memory_manager.stores) == {
        "working",
        "semantic",
        "episodic",
        "perceptual",
    }


def test_http_dashboard_can_switch_to_persistent_builder(tmp_path, monkeypatch):
    monkeypatch.setenv("CHAPTER8_DASHBOARD_PERSISTENT", "1")
    monkeypatch.setenv("CHAPTER8_DASHBOARD_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("CHAPTER8_DASHBOARD_REAL_LLM", "0")
    monkeypatch.delenv("CHAPTER8_DASHBOARD_EXTERNAL_BACKENDS", raising=False)

    dashboard = build_http_dashboard()

    assert dashboard.rag_tool.document_store.path == tmp_path / "rag_documents.sqlite3"
    assert dashboard.rag_tool.backend_mode == "local_persistent"
    assert dashboard.backend_status()["backend_mode"] == "sqlite"


def test_http_dashboard_can_enable_cross_modal_embedding_entrypoint(tmp_path, monkeypatch):
    monkeypatch.setenv("CHAPTER8_DASHBOARD_PERSISTENT", "1")
    monkeypatch.setenv("CHAPTER8_DASHBOARD_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("CHAPTER8_DASHBOARD_REAL_LLM", "0")
    monkeypatch.setenv("CHAPTER8_DASHBOARD_CROSS_MODAL_EMBEDDINGS", "1")

    dashboard = build_http_dashboard()

    assert {"image", "audio"} <= set(dashboard.perception_tool.encoders)


def test_memory_rag_dashboard_exposes_cross_modal_file_search(tmp_path):
    chart_a = tmp_path / "chart_a.png"
    chart_b = tmp_path / "chart_b.png"
    query_chart = tmp_path / "query_chart.png"
    chart_a.write_bytes(b"chart a")
    chart_b.write_bytes(b"chart b")
    query_chart.write_bytes(b"query chart")
    vectors = {
        chart_a.name: [1.0, 0.0],
        chart_b.name: [0.0, 1.0],
        query_chart.name: [0.9, 0.1],
    }
    manager = MyMemoryManager(stores={"perceptual": PerceptualMemoryStore()})
    tool = MyPerceptionTool(
        manager=manager,
        encoders={"image": lambda path: vectors[path.name]},
    )
    dashboard = MemoryRAGDashboard(perception_tool=tool, memory_manager=manager)

    dashboard.ingest_file(chart_a, description="支付流程架构图")
    dashboard.ingest_file(chart_b, description="音频处理流程图")
    result = dashboard.search_similar_file(query_chart, limit=1)

    assert "Found 1 perceptual memories by file embedding" in result
    assert "支付流程架构图" in result
    assert "音频处理流程图" not in result


def test_persistent_dashboard_reports_rag_document_inventory(tmp_path):
    note = tmp_path / "chapter8_rag_note.md"
    note.write_text(
        "# 第八章\n\n## RAG\n\nRAG 包括文档导入、切块、检索和生成。",
        encoding="utf-8",
    )
    dashboard = build_persistent_dashboard_demo(data_dir=tmp_path, strict_backends=False)

    dashboard.load_document(note)
    inventory = dashboard.rag_inventory()
    payload = json.loads(inventory)

    assert "chapter8_rag_note.md" in inventory
    assert payload["document_count"] == 1
    assert payload["documents"][0]["chunk_count"] == len(
        dashboard.rag_tool.document_store.list_chunks(
            "chapter8_rag_note.md",
            namespace="business_multimodal",
        )
    )
    assert '"parser": "plain_text"' in inventory
    assert payload["documents"][0]["source_path"] == str(note)


def test_persistent_dashboard_can_delete_rag_document(tmp_path):
    note = tmp_path / "chapter8_rag_note.md"
    note.write_text("RAG 删除文档时要同步清理 chunk 元数据和向量索引。", encoding="utf-8")
    dashboard = build_persistent_dashboard_demo(data_dir=tmp_path, strict_backends=False)
    dashboard.load_document(note)

    delete_output = dashboard.delete_document("chapter8_rag_note.md")
    payload = json.loads(dashboard.rag_inventory())

    assert "文档已删除: chapter8_rag_note.md" in delete_output
    assert payload["document_count"] == 0
    assert "chapter8_rag_note.md" not in dashboard.ask("RAG 删除文档", limit=3)


def test_dashboard_cascade_deletes_perceptual_document_memories_and_trace(tmp_path):
    dashboard = build_dashboard_demo(
        prefer_real_multimodal=True,
        image_ocr=lambda path: "图片里写着级联删除 CASCADE-IMG-001",
    )
    image = tmp_path / "cascade.png"
    image.write_bytes(b"fake image")

    ingest_output = dashboard.ingest_file(image, description="级联删除测试图片")
    document_id = "perceptual:image:cascade.png"

    assert "CASCADE-IMG-001" in ingest_output
    assert "CASCADE-IMG-001" in dashboard.ask("CASCADE-IMG-001 是什么？")
    assert "CASCADE-IMG-001" in dashboard.recall("CASCADE-IMG-001")
    assert "CASCADE-IMG-001" in dashboard.trace()

    delete_output = dashboard.delete_document(document_id)

    assert "文档已删除" in delete_output
    assert "关联记忆删除数" in delete_output
    assert "CASCADE-IMG-001" not in dashboard.trace()
    answer_after_delete = dashboard.ask("CASCADE-IMG-001 是什么？")
    assert "未找到" in answer_after_delete
    assert dashboard.rag_tool.last_retrieved_chunks == []
    assert "CASCADE-IMG-001" not in dashboard.recall("CASCADE-IMG-001")


def test_http_upload_job_status_formats_running_and_completed_states(tmp_path):
    handler = DashboardHTTPHandler.__new__(DashboardHTTPHandler)
    job_id = "abc123"
    DashboardHTTPHandler._set_upload_job(
        job_id,
        {
            "status": "running",
            "file_path": str(tmp_path / "upload.png"),
            "created_at": 100.0,
            "updated_at": 100.0,
            "result": "",
            "error": "",
        },
    )

    running = handler._format_upload_job(job_id)
    DashboardHTTPHandler._update_upload_job(
        job_id,
        status="completed",
        result="Perceptual memory saved:\n- rag_sync: skipped (empty extracted_text)",
    )
    completed = handler._format_upload_job(job_id)

    assert "status: running" in running
    assert "后台正在执行 OCR/ASR" in running
    assert "status: completed" in completed
    assert "Perceptual memory saved" in completed
