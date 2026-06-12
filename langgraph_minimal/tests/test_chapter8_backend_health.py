import sys

from scripts.chapter8_backend_health import build_health_report


def test_chapter8_backend_health_reports_local_models_and_masks_secrets(tmp_path, monkeypatch):
    model_root = tmp_path / "models"
    (model_root / "all-MiniLM-L6-v2").mkdir(parents=True)
    (model_root / "all-MiniLM-L6-v2" / "model.safetensors").write_text("x", encoding="utf-8")
    (model_root / "iic" / "SenseVoiceSmall").mkdir(parents=True)
    (model_root / "iic" / "SenseVoiceSmall" / "model.pt").write_text("x", encoding="utf-8")
    (model_root / "iic" / "SenseVoiceSmall" / "config.yaml").write_text("x", encoding="utf-8")
    (model_root / "PaddlePaddle" / "PaddleOCR-VL-1.6").mkdir(parents=True)
    (model_root / "PaddlePaddle" / "PaddleOCR-VL-1.6" / "model.safetensors").write_text("x", encoding="utf-8")
    (model_root / "PaddlePaddle" / "PaddleOCR-VL-1.6" / "config.json").write_text("x", encoding="utf-8")

    monkeypatch.setenv("QDRANT_URL", "http://127.0.0.1:6333")
    monkeypatch.setenv("NEO4J_URI", "neo4j+s://example.databases.neo4j.io")
    monkeypatch.setenv("NEO4J_USERNAME", "neo4j-user")
    monkeypatch.setenv("NEO4J_PASSWORD", "super-secret-password")
    monkeypatch.setenv("EMBED_MODEL_NAME", str(model_root / "all-MiniLM-L6-v2"))

    report = build_health_report(model_root=model_root, check_services=False)

    assert report["models"]["text_embedding"]["ready"] is True
    assert report["models"]["sensevoice_asr"]["ready"] is True
    assert report["models"]["paddleocr_vl"]["ready"] is True
    assert report["services"]["qdrant"]["configured"] is True
    assert report["services"]["neo4j"]["configured"] is True
    assert report["services"]["neo4j"]["password"] == "***"
    assert "super-secret-password" not in str(report)


def test_chapter8_backend_health_accepts_external_multimodal_runtime(tmp_path, monkeypatch):
    model_root = tmp_path / "models"
    (model_root / "all-MiniLM-L6-v2").mkdir(parents=True)
    (model_root / "all-MiniLM-L6-v2" / "model.safetensors").write_text("x", encoding="utf-8")
    (model_root / "iic" / "SenseVoiceSmall").mkdir(parents=True)
    (model_root / "iic" / "SenseVoiceSmall" / "model.pt").write_text("x", encoding="utf-8")
    (model_root / "iic" / "SenseVoiceSmall" / "config.yaml").write_text("x", encoding="utf-8")
    (model_root / "PaddlePaddle" / "PaddleOCR-VL-1.6").mkdir(parents=True)
    (model_root / "PaddlePaddle" / "PaddleOCR-VL-1.6" / "model.safetensors").write_text("x", encoding="utf-8")
    (model_root / "PaddlePaddle" / "PaddleOCR-VL-1.6" / "config.json").write_text("x", encoding="utf-8")

    fake_site = tmp_path / "fake_site"
    (fake_site / "funasr").mkdir(parents=True)
    (fake_site / "funasr" / "__init__.py").write_text("", encoding="utf-8")
    (fake_site / "paddleocr").mkdir(parents=True)
    (fake_site / "paddleocr" / "__init__.py").write_text("", encoding="utf-8")

    monkeypatch.setenv("PYTHONPATH", str(fake_site))
    monkeypatch.setenv("MULTIMODAL_PYTHON", sys.executable)
    monkeypatch.setenv("EMBED_MODEL_NAME", str(model_root / "all-MiniLM-L6-v2"))

    report = build_health_report(model_root=model_root, check_services=False)

    runtime = report["multimodal_runtime"]
    assert runtime["configured"] is True
    assert runtime["packages"]["funasr"]["installed"] is True
    assert runtime["packages"]["paddleocr"]["installed"] is True
    assert report["ready"] is True
