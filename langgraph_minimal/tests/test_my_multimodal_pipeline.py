from pathlib import Path
import sys

from app.my_memory_system import MyMemoryManager, PerceptualMemoryStore
from app.my_memory_system.multimodal_pipeline import (
    ExternalRuntimeASR,
    ExternalRuntimeOCR,
    build_multimodal_perception_tool,
)
from scripts.chapter8_multimodal_pipeline_demo import build_sample_files, run_demo


class FakeRAGTool:
    def __init__(self):
        self.calls = []

    def run(self, parameters):
        self.calls.append(parameters)
        if parameters["action"] == "add_text":
            return f"fake rag added {parameters['document_id']}"
        if parameters["action"] == "search":
            return "搜索结果：fake rag hit"
        return "fake rag ok"


def test_build_multimodal_perception_tool_wires_all_processors(tmp_path):
    manager = MyMemoryManager(
        user_id="pipeline_user",
        stores={"perceptual": PerceptualMemoryStore()},
    )
    rag_tool = FakeRAGTool()
    image_path = tmp_path / "diagram.png"
    audio_path = tmp_path / "meeting.wav"
    image_path.write_bytes(b"fake image")
    audio_path.write_bytes(b"fake audio")

    tool = build_multimodal_perception_tool(
        manager=manager,
        rag_tool=rag_tool,
        image_ocr=lambda path: "图片里写着 Qdrant 与 CLIP",
        audio_asr=lambda path: "音频里提到 ASR 与 CLAP",
        image_encoder=lambda path: [1.0, 0.0],
        audio_encoder=lambda path: [0.0, 1.0],
        rag_namespace="pipeline_test",
    )

    image_result = tool.run({"action": "ingest_file", "file_path": str(image_path)})
    audio_result = tool.run({"action": "ingest_file", "file_path": str(audio_path)})

    assert "modality: image" in image_result
    assert "embedding_dim: 2" in image_result
    assert "Qdrant" in image_result
    assert "modality: audio" in audio_result
    assert "CLAP" in audio_result
    assert len(manager.stores["perceptual"].records) == 2
    assert [call["namespace"] for call in rag_tool.calls] == ["pipeline_test", "pipeline_test"]


def test_chapter8_multimodal_pipeline_demo_runs_end_to_end(tmp_path):
    files = build_sample_files(tmp_path)
    result = run_demo(tmp_path, files=files)

    assert "Directory perceptual ingest finished" in result.ingest_result
    assert "Found" in result.memory_search_result
    assert "搜索结果" in result.rag_search_result
    assert len(result.records) == 3
    assert any(event["stage"] == "perception.sync_rag" for event in result.trace_events)
    assert any(record.metadata.get("embedding_dim") for record in result.records)


def test_external_runtime_adapters_call_configured_python_worker(tmp_path):
    worker = tmp_path / "worker.py"
    worker.write_text(
        """
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("action")
parser.add_argument("--file")
parser.add_argument("--model-dir")
args = parser.parse_args()
print(json.dumps({"text": f"{args.action}:{args.file}:{args.model_dir}"}, ensure_ascii=False))
""".strip(),
        encoding="utf-8",
    )
    media_file = tmp_path / "sample.wav"
    media_file.write_bytes(b"fake media")

    asr = ExternalRuntimeASR(
        python_path=sys.executable,
        worker_path=worker,
        model_dir=tmp_path / "asr_model",
    )
    ocr = ExternalRuntimeOCR(
        python_path=sys.executable,
        worker_path=worker,
        model_dir=tmp_path / "ocr_model",
    )

    assert "asr:" in asr(media_file)
    assert str(tmp_path / "asr_model") in asr(media_file)
    assert "ocr:" in ocr(media_file)
    assert str(tmp_path / "ocr_model") in ocr(media_file)


def test_build_multimodal_tool_can_use_external_runtime_worker(tmp_path, monkeypatch):
    monkeypatch.setenv("CHAPTER8_FAKE_MULTIMODAL", "1")
    manager = MyMemoryManager(
        user_id="external_runtime_user",
        stores={"perceptual": PerceptualMemoryStore()},
    )
    image_path = tmp_path / "diagram.png"
    audio_path = tmp_path / "meeting.wav"
    image_path.write_bytes(b"fake image")
    audio_path.write_bytes(b"fake audio")

    tool = build_multimodal_perception_tool(
        manager=manager,
        enable_ocr=True,
        enable_asr=True,
        enable_image_embedding=False,
        enable_audio_embedding=False,
        enable_rag=False,
        multimodal_python=sys.executable,
        multimodal_worker=Path("scripts") / "chapter8_multimodal_worker.py",
        model_root=tmp_path / "models",
    )

    image_result = tool.run({"action": "ingest_file", "file_path": str(image_path)})
    audio_result = tool.run({"action": "ingest_file", "file_path": str(audio_path)})

    assert "extractor: image_ocr" in image_result
    assert "fake OCR text" in image_result
    assert "extractor: audio_asr" in audio_result
    assert "fake ASR text" in audio_result
    assert len(manager.stores["perceptual"].records) == 2
