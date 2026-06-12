from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .rag_tool import MyRAGTool


@dataclass
class RAGQAResult:
    """Structured result for the chapter 8 style document QA demo."""

    answer: str
    references: list[str]
    retrieved_chunks: list[dict[str, Any]]
    trace: list[dict[str, Any]]
    raw_output: str


class RAGQADemo:
    """End-to-end document QA workflow built on MyRAGTool.

    第八章的 RAG 应用可以拆成四步：文档导入、检索、基于上下文回答、
    展示引用与 trace。这个类把这些步骤串起来，便于脚本、测试或后续
    Gradio 页面复用，而不是把流程散落在临时脚本里。
    """

    def __init__(
        self,
        *,
        rag_tool: MyRAGTool | None = None,
        namespace: str = "chapter8_demo",
    ) -> None:
        self.rag_tool = rag_tool or MyRAGTool()
        self.namespace = namespace

    def ingest_document(
        self,
        file_path: str | Path,
        *,
        document_id: str | None = None,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
    ) -> str:
        return self.rag_tool.run(
            {
                "action": "add_document",
                "file_path": str(file_path),
                "document_id": document_id,
                "namespace": self.namespace,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
            }
        )

    def ingest_text(
        self,
        text: str,
        *,
        document_id: str,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
    ) -> str:
        return self.rag_tool.run(
            {
                "action": "add_text",
                "text": text,
                "document_id": document_id,
                "namespace": self.namespace,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
            }
        )

    def ask(
        self,
        question: str,
        *,
        limit: int = 5,
        enable_mqe: bool = True,
        enable_hyde: bool = False,
        score_threshold: float | None = None,
    ) -> RAGQAResult:
        output = self.rag_tool.run(
            {
                "action": "ask",
                "question": question,
                "namespace": self.namespace,
                "limit": limit,
                "enable_mqe": enable_mqe,
                "enable_hyde": enable_hyde,
                "score_threshold": score_threshold,
            }
        )
        return RAGQAResult(
            answer=self._extract_answer(output),
            references=self._extract_references(output),
            retrieved_chunks=list(self.rag_tool.last_retrieved_chunks),
            trace=list(self.rag_tool.trace_events),
            raw_output=output,
        )

    def _extract_answer(self, output: str) -> str:
        if "参考来源：" not in output:
            return output.strip()
        answer_part = output.split("参考来源：", 1)[0]
        return answer_part.replace("智能问答结果：", "", 1).strip()

    def _extract_references(self, output: str) -> list[str]:
        if "参考来源：" not in output:
            return []
        reference_part = output.split("参考来源：", 1)[1]
        return [
            line.strip()
            for line in reference_part.splitlines()
            if line.strip()
        ]
