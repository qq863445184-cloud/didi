from scripts.chapter8_memory_rag_dashboard_demo import (
    build_dashboard_demo,
    build_persistent_dashboard_demo,
)
from scripts.chapter8_memory_rag_dashboard_http_demo import build_http_dashboard


def test_build_dashboard_demo_wires_business_multimodal_stack():
    dashboard = build_dashboard_demo()

    assert dashboard.rag_tool is not None
    assert dashboard.perception_tool is not None
    assert dashboard.memory_manager is not None
    assert dashboard.rag_namespace == "business_multimodal"


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
