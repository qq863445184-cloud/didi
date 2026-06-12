from __future__ import annotations

import math
import struct
import sys
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.my_memory_system import MyMemoryManager, PerceptualMemoryStore
from app.my_memory_system.multimodal_pipeline import build_multimodal_perception_tool


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


class DemoEmbedder:
    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            if any(keyword in text for keyword in ("OCR", "CLIP", "图片", "架构图")):
                vectors.append([1.0, 0.0, 0.0])
            elif any(keyword in text for keyword in ("ASR", "CLAP", "音频", "会议")):
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])
        return vectors


class DemoVectorStore:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def add_vectors(self, vectors, metadata, ids=None):
        ids = ids or [str(index) for index in range(len(vectors))]
        for vector, meta, row_id in zip(vectors, metadata, ids):
            self.rows.append({"id": row_id, "vector": vector, "metadata": meta})
        return True

    def search_similar(self, query_vector, limit=5, score_threshold=None, where=None):
        hits = []
        for row in self.rows:
            if where and not all(row["metadata"].get(key) == value for key, value in where.items()):
                continue
            score = sum(a * b for a, b in zip(query_vector, row["vector"]))
            if score_threshold is not None and score < score_threshold:
                continue
            hits.append({"id": row["id"], "score": score, "metadata": row["metadata"]})
        hits.sort(key=lambda item: item["score"], reverse=True)
        return hits[:limit]


class DemoLLM:
    def invoke(self, messages, **kwargs):
        return "多模态流水线已经把文本、图片 OCR 和音频 ASR 同步到了 RAG。"


@dataclass
class MultimodalPipelineDemoResult:
    ingest_result: str
    memory_search_result: str
    rag_search_result: str
    file_search_result: str
    records: list[Any]
    trace_events: list[dict[str, Any]]


def build_sample_files(directory: str | Path) -> list[Path]:
    """Create one text, one image, and one audio file for a deterministic demo."""

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    note_path = directory / "agent_memory_note.md"
    note_path.write_text(
        "第八章的多模态记忆系统包含感知记忆、RAG 检索、图片 OCR 和音频 ASR。",
        encoding="utf-8",
    )

    image_path = directory / "architecture_diagram.png"
    try:
        from PIL import Image

        Image.new("RGB", (32, 32), (80, 120, 200)).save(image_path)
    except Exception:
        image_path.write_bytes(b"fake image bytes")

    audio_path = directory / "meeting.wav"
    _write_tone_wav(audio_path)
    return [note_path, image_path, audio_path]


def run_demo(
    directory: str | Path,
    *,
    files: list[Path] | None = None,
) -> MultimodalPipelineDemoResult:
    """Run an end-to-end multimodal perception pipeline without external services."""

    directory = Path(directory)
    files = files or build_sample_files(directory)
    manager = MyMemoryManager(
        user_id="chapter8_multimodal_demo",
        stores={"perceptual": PerceptualMemoryStore()},
    )
    from app.my_memory_system import MyRAGTool

    rag_tool = MyRAGTool(
        embedder=DemoEmbedder(),
        vector_store=DemoVectorStore(),
        llm=DemoLLM(),
        collection_name="chapter8_multimodal_demo",
    )
    tool = build_multimodal_perception_tool(
        manager=manager,
        rag_tool=rag_tool,
        rag_namespace="chapter8_multimodal_demo",
        image_ocr=lambda path: "架构图展示 OCR、CLIP、感知记忆和 RAG 的连接关系。",
        audio_asr=lambda path: "会议音频讨论了 ASR、CLAP、Qdrant 和多模态检索。",
        image_encoder=lambda path: [1.0, 0.0, 0.0],
        audio_encoder=lambda path: [0.0, 1.0, 0.0],
    )

    ingest_result = tool.run(
        {
            "action": "ingest_directory",
            "directory_path": str(directory),
            "importance": 0.8,
            "description": "第八章多模态流水线演示资料。",
        }
    )
    memory_search_result = tool.run(
        {"action": "search", "query": "OCR ASR CLIP CLAP RAG", "limit": 5}
    )
    rag_search_result = rag_tool.run(
        {
            "action": "search",
            "query": "图片 OCR 和音频 ASR 如何进入 RAG",
            "namespace": "chapter8_multimodal_demo",
            "limit": 5,
        }
    )
    file_search_result = tool.run(
        {
            "action": "search_file",
            "file_path": str(next(path for path in files if path.suffix == ".png")),
            "limit": 2,
        }
    )

    return MultimodalPipelineDemoResult(
        ingest_result=ingest_result,
        memory_search_result=memory_search_result,
        rag_search_result=rag_search_result,
        file_search_result=file_search_result,
        records=manager.stores["perceptual"].records,
        trace_events=tool.trace_events,
    )


def _write_tone_wav(path: Path) -> None:
    sample_rate = 16000
    frames = sample_rate // 4
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(
            b"".join(
                struct.pack(
                    "<h",
                    int(12000 * math.sin(2 * math.pi * 440 * index / sample_rate)),
                )
                for index in range(frames)
            )
        )


def main() -> None:
    result = run_demo(PROJECT_ROOT / "memory_data" / "multimodal_demo")
    print("=== ingest ===")
    print(result.ingest_result)
    print("\n=== memory search ===")
    print(result.memory_search_result)
    print("\n=== rag search ===")
    print(result.rag_search_result)
    print("\n=== file embedding search ===")
    print(result.file_search_result)
    print("\n=== trace ===")
    for event in result.trace_events:
        print(event)


if __name__ == "__main__":
    main()
