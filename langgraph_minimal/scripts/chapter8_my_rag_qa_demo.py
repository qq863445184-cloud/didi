from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.my_memory_system import MyRAGTool, RAGQADemo


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


class DemoEmbedder:
    """Deterministic embedder so this demo can run without external services."""

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            if "RAG" in text or "检索" in text or "知识库" in text:
                vectors.append([1.0, 0.0, 0.0])
            elif "记忆" in text or "工作记忆" in text or "语义记忆" in text:
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])
        return vectors


class DemoVectorStore:
    """Tiny in-memory vector store with the same methods MyRAGTool needs."""

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
    """Small fake LLM that keeps the demo focused on RAG orchestration."""

    def invoke(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        prompt = messages[-1]["content"]
        if "改写成 3 个" in prompt:
            return "RAG 检索流程\n知识库问答\n引用展示"
        return "RAG 问答通常包括文档导入、切块入库、向量检索、基于上下文生成答案和展示引用。"


def build_demo() -> RAGQADemo:
    rag_tool = MyRAGTool(
        embedder=DemoEmbedder(),
        vector_store=DemoVectorStore(),
        llm=DemoLLM(),
        collection_name="chapter8_demo_memory",
    )
    return RAGQADemo(rag_tool=rag_tool, namespace="chapter8_demo")


def build_sample_document(directory: Path) -> Path:
    document_path = directory / "chapter8_rag_note.md"
    document_path.write_text(
        "\n\n".join(
            [
                "# 第八章：记忆与检索",
                "工作记忆保存当前任务上下文，语义记忆保存稳定知识，情景记忆保存具体事件。",
                "## RAG 问答",
                "RAG 问答通常包括文档导入、切块入库、向量检索、基于上下文生成答案和展示引用。",
            ]
        ),
        encoding="utf-8",
    )
    return document_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Chapter 8 custom RAG QA demo")
    parser.add_argument("--question", default="RAG 问答流程包括什么？")
    args = parser.parse_args()

    demo = build_demo()
    with tempfile.TemporaryDirectory() as temp_dir:
        document_path = build_sample_document(Path(temp_dir))
        print("[1] ingest document")
        print(demo.ingest_document(document_path, chunk_size=48, chunk_overlap=8))

        print("\n[2] ask")
        result = demo.ask(args.question, enable_mqe=True, limit=3)
        print(result.answer)

        print("\n[3] references")
        for reference in result.references:
            print(reference)

        print("\n[4] trace")
        for event in result.trace:
            print(event)


if __name__ == "__main__":
    main()
