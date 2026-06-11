from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.my_memory_system import DocumentLearningAssistant, MyMemoryManager, MyRAGTool, WorkingMemoryStore


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


class DemoEmbedder:
    """Deterministic embedder so the demo does not need Qdrant/online embeddings."""

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            if "RAG" in text or "检索" in text or "知识库" in text:
                vectors.append([1.0, 0.0, 0.0])
            elif "记忆" in text or "学习" in text:
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])
        return vectors


class DemoVectorStore:
    """Minimal in-memory vector store implementing the MyRAGTool contract."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def add_vectors(
        self,
        vectors: list[list[float]],
        metadata: list[dict[str, Any]],
        ids: list[str] | None = None,
    ) -> bool:
        row_ids = ids or [str(index) for index in range(len(vectors))]
        for vector, meta, row_id in zip(vectors, metadata, row_ids):
            self.rows.append({"id": row_id, "vector": vector, "metadata": meta})
        return True

    def search_similar(
        self,
        query_vector: list[float],
        limit: int = 5,
        score_threshold: float | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        scored: list[dict[str, Any]] = []
        for row in self.rows:
            if where and not all(row["metadata"].get(key) == value for key, value in where.items()):
                continue
            score = sum(left * right for left, right in zip(query_vector, row["vector"]))
            if score_threshold is not None and score < score_threshold:
                continue
            scored.append({"id": row["id"], "score": score, "metadata": row["metadata"]})
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:limit]


class DemoLLM:
    """Fake LLM focused on proving orchestration rather than model quality."""

    def invoke(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        prompt = messages[-1]["content"]
        if "改写成 3 个" in prompt:
            return "RAG 学习流程\n知识库证据\n学习记忆"
        return "RAG 学习流程包括文档入库、语义检索、基于上下文回答、引用追踪，并把学习过程写入记忆。"


def build_demo() -> DocumentLearningAssistant:
    rag_tool = MyRAGTool(
        embedder=DemoEmbedder(),
        vector_store=DemoVectorStore(),
        llm=DemoLLM(),
        collection_name="chapter8_document_learning_demo",
    )
    manager = MyMemoryManager(
        user_id="chapter8_learner",
        stores={
            # Demo 中四类记忆都用轻量本地 store，便于无服务环境直接运行。
            "working": WorkingMemoryStore(),
            "semantic": WorkingMemoryStore(),
            "episodic": WorkingMemoryStore(),
            "perceptual": WorkingMemoryStore(),
        },
    )
    return DocumentLearningAssistant(
        rag_tool=rag_tool,
        memory_manager=manager,
        namespace="chapter8_document_learning",
        session_id="demo-session",
    )


def build_sample_document(directory: Path) -> Path:
    document_path = directory / "chapter8_learning_note.md"
    document_path.write_text(
        "\n\n".join(
            [
                "# 第八章：文档学习助手",
                "RAG 学习流程包括文档入库、切块、语义检索、基于上下文回答和引用追踪。",
                "## 记忆闭环",
                "学习助手还会把加载文档、提出问题、学习笔记等过程写入工作记忆、语义记忆和情景记忆。",
            ]
        ),
        encoding="utf-8",
    )
    return document_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Chapter 8 RAG + Memory learning assistant demo")
    parser.add_argument("--question", default="RAG 学习流程包括什么？")
    parser.add_argument("--note", default="RAG 学习要同时关注知识库证据和学习记忆。")
    args = parser.parse_args()

    assistant = build_demo()
    with tempfile.TemporaryDirectory() as temp_dir:
        document_path = build_sample_document(Path(temp_dir))

        print("[1] load document")
        print(assistant.load_document(document_path, chunk_size=48, chunk_overlap=8).raw_output)

        print("\n[2] ask")
        answer = assistant.ask(args.question, limit=3)
        print(answer.answer)

        print("\n[3] add note")
        print(assistant.add_note(args.note))

        print("\n[4] recall")
        print(assistant.recall(args.note, limit=3))

        print("\n[5] learning report")
        print(assistant.generate_report())

        print("\n[6] trace")
        for event in assistant.trace_events:
            print(event)


if __name__ == "__main__":
    main()
