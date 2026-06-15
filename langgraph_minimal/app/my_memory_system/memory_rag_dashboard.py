from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .document_learning_assistant import DocumentLearningAssistant
from .manager import MyMemoryManager
from .perception_tool import MyPerceptionTool
from .rag_tool import MyRAGTool


class MemoryRAGDashboard:
    """Productized callback layer for the chapter 8 memory/RAG system.

    这一层面向“页面交互”：文档加载、多模态感知、RAG 问答、记忆检索、
    系统状态和 trace 都从一个对象暴露出去。底层能力仍然由 assistant、
    perception_tool、rag_tool 和 memory_manager 承担，避免 UI 直接耦合存储细节。
    """

    def __init__(
        self,
        *,
        assistant: DocumentLearningAssistant | None = None,
        perception_tool: MyPerceptionTool | None = None,
        rag_tool: MyRAGTool | None = None,
        memory_manager: MyMemoryManager | None = None,
        rag_namespace: str = "chapter8_dashboard",
    ) -> None:
        self.assistant = assistant
        self.perception_tool = perception_tool
        self.rag_tool = rag_tool or getattr(assistant, "rag_tool", None)
        self.memory_manager = (
            memory_manager
            or getattr(assistant, "memory_manager", None)
            or getattr(perception_tool, "manager", None)
        )
        self.rag_namespace = rag_namespace

    def load_document(self, file_path: str | Path | None) -> str:
        if not file_path:
            return "请先选择文档。"
        if self.assistant is not None:
            return self.assistant.load_document(file_path).raw_output
        if self.rag_tool is None:
            return "未配置 RAG 工具，无法加载文档。"
        return self.rag_tool.run(
            {
                "action": "add_document",
                "file_path": str(file_path),
                "namespace": self.rag_namespace,
            }
        )

    def ingest_file(
        self,
        file_path: str | Path | None,
        description: str = "",
        importance: float = 0.7,
    ) -> str:
        if not file_path:
            return "请先选择要感知入库的文件。"
        if self.perception_tool is None:
            return "未配置感知工具，无法处理多模态文件。"
        return self.perception_tool.run(
            {
                "action": "ingest_file",
                "file_path": str(file_path),
                "description": description,
                "importance": importance,
            }
        )

    def ask(
        self,
        question: str,
        *,
        limit: int = 5,
        enable_mqe: bool = False,
        enable_hyde: bool = False,
        enable_keyword_rerank: bool = True,
    ) -> str:
        normalized = question.strip()
        if not normalized:
            return "请输入问题。"
        if self.rag_tool is None:
            return "未配置 RAG 工具，无法问答。"

        search_output = self.rag_tool.run(
            {
                "action": "search",
                "query": normalized,
                "namespace": self.rag_namespace,
                "limit": limit,
                "enable_mqe": enable_mqe,
                "enable_hyde": enable_hyde,
                "enable_keyword_rerank": enable_keyword_rerank,
            }
        )
        chunks = list(self.rag_tool.last_retrieved_chunks)
        if not chunks:
            return search_output

        useful_chunks = self._select_useful_chunks(normalized, chunks)
        raw_answer = self._build_grounded_answer(normalized, useful_chunks)
        evidence = self._format_retrieved_chunks(useful_chunks)
        if not evidence:
            return raw_answer
        return "\n\n".join([raw_answer, "检索证据：", evidence])

    def recall(self, query: str, *, limit: int = 5) -> str:
        normalized = query.strip()
        if not normalized:
            return "请输入要检索的记忆关键词。"
        if self.memory_manager is None:
            return "未配置记忆管理器，无法检索记忆。"
        results = self.memory_manager.search(
            query=normalized,
            memory_type="all",
            limit=limit,
        )
        if not results:
            return "未找到相关记忆。"
        lines = ["记忆检索结果："]
        for index, item in enumerate(results, 1):
            lines.append(
                f"{index}. type={item.record.memory_type} score={item.score:.3f} "
                f"importance={item.record.importance:.2f}\n"
                f"   {item.record.content}"
            )
        return "\n".join(lines)

    def memory_inventory(self) -> str:
        if self.memory_manager is not None:
            payload = self.memory_manager.stats()
        elif self.assistant is not None:
            payload = self.assistant.get_stats()
        else:
            payload = {"total": 0, "by_type": {}}
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def trace(self) -> str:
        return json.dumps(self._trace_events(), ensure_ascii=False, indent=2)

    def _trace_events(self) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for source in (self.perception_tool, self.rag_tool, self.memory_manager, self.assistant):
            events.extend(list(getattr(source, "trace_events", []) or []))
        return events

    def _format_retrieved_chunks(self, chunks: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for index, chunk in enumerate(chunks, 1):
            content = str(chunk.get("content", "")).replace("\n", " ")[:240]
            lines.append(
                f"{index}. document={chunk.get('document_id')} "
                f"chunk={chunk.get('chunk_index')} "
                f"score={float(chunk.get('score', 0.0)):.3f} "
                f"vector={float(chunk.get('vector_score', chunk.get('score', 0.0))):.3f} "
                f"keyword={float(chunk.get('keyword_score', 0.0)):.3f}\n"
                f"   {content}"
            )
        return "\n".join(lines)

    def _build_grounded_answer(self, question: str, chunks: list[dict[str, Any]]) -> str:
        """Build a transparent answer from retrieved chunks for the lightweight UI.

        The dashboard demo often uses a deterministic fake LLM so it can run without
        external services.  That fake model is useful for tests, but too rigid for
        manual uploads.  Here we synthesize an extractive answer directly from the
        retrieved evidence, so the page demonstrates true RAG grounding even when
        no real generation model is configured.
        """

        if not chunks:
            return f"智能问答结果：\n未找到与“{question}”足够相关的知识库内容。"

        lines = [
            "智能问答结果：",
            f"根据检索到的 {len(chunks)} 个高相关片段，可以回答如下：",
        ]
        for index, chunk in enumerate(chunks, 1):
            content = self._clean_chunk_content(str(chunk.get("content", "")))
            lines.append(
                f"{index}. {content}"
            )
        lines.extend(
            [
                "",
                "参考来源：",
            ]
        )
        for index, chunk in enumerate(chunks, 1):
            lines.append(
                f"{index}. {chunk.get('document_id')}#chunk-{chunk.get('chunk_index')} "
                f"score={float(chunk.get('score', 0.0)):.3f}"
            )
        return "\n".join(lines)

    def _select_useful_chunks(
        self,
        question: str,
        chunks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Pick evidence that is relevant to the current question.

        The demo can mix deterministic OCR/ASR fixtures with user uploads.  Those
        fixtures may have high vector scores even when the question is about a
        newly uploaded document, so direct keyword matches are treated as stronger
        evidence for the dashboard answer.  If no keyword match exists, we fall
        back to the vector/rerank order from the RAG tool.
        """

        positive = [
            chunk
            for chunk in chunks
            if float(chunk.get("score", 0.0)) > 0.0
            or float(chunk.get("keyword_score", 0.0)) > 0.0
            or float(chunk.get("vector_score", 0.0)) > 0.0
        ]
        keyword_hits = [
            chunk
            for chunk in positive
            if float(chunk.get("keyword_score", 0.0)) > 0.0
        ]
        candidates = keyword_hits or positive or chunks
        return candidates[:3]

    def _clean_chunk_content(self, content: str, *, max_chars: int = 360) -> str:
        raw = content.strip()
        if raw.startswith("# Perceptual source:") and "\n\n" in raw:
            # Perceptual RAG chunks prepend source metadata before extracted text.
            raw = raw.split("\n\n", 1)[1].strip()
        normalized = " ".join(raw.split())
        if len(normalized) <= max_chars:
            return normalized
        return f"{normalized[:max_chars].rstrip()}..."


def build_memory_rag_dashboard_app(dashboard: MemoryRAGDashboard) -> Any:
    """Build the optional Gradio page for product-style manual testing."""

    try:
        import gradio as gr
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Gradio 未安装。请先执行 pip install gradio 后再启动 Memory/RAG 管理页。"
        ) from exc

    with gr.Blocks(title="第八章 Memory/RAG 管理页") as demo:
        gr.Markdown("# 第八章 Memory/RAG 管理页")
        with gr.Tab("文档入库"):
            document_file = gr.File(label="上传文档", type="filepath")
            document_output = gr.Textbox(label="入库结果", lines=8)
            gr.Button("加载文档").click(
                dashboard.load_document,
                inputs=document_file,
                outputs=document_output,
            )

        with gr.Tab("多模态感知"):
            perceptual_file = gr.File(label="上传图片/音频/视频/文本", type="filepath")
            description = gr.Textbox(label="描述")
            importance = gr.Slider(label="重要性", minimum=0.0, maximum=1.0, value=0.7)
            perceptual_output = gr.Textbox(label="感知入库结果", lines=8)
            gr.Button("感知入库").click(
                dashboard.ingest_file,
                inputs=[perceptual_file, description, importance],
                outputs=perceptual_output,
            )

        with gr.Tab("RAG 问答"):
            question = gr.Textbox(label="问题")
            answer = gr.Textbox(label="回答与证据", lines=14)
            gr.Button("提问").click(dashboard.ask, inputs=question, outputs=answer)

        with gr.Tab("记忆管理"):
            recall_query = gr.Textbox(label="记忆检索关键词")
            recall_output = gr.Textbox(label="记忆检索结果", lines=10)
            inventory_output = gr.Code(label="记忆库存", language="json")
            gr.Button("检索记忆").click(dashboard.recall, inputs=recall_query, outputs=recall_output)
            gr.Button("刷新库存").click(dashboard.memory_inventory, outputs=inventory_output)

        with gr.Tab("Trace"):
            trace_output = gr.Code(label="Trace", language="json")
            gr.Button("刷新 Trace").click(dashboard.trace, outputs=trace_output)

    return demo
