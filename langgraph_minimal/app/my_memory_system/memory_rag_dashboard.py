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

        generated_answer = self.rag_tool.run(
            {
                "action": "ask",
                "question": normalized,
                "namespace": self.rag_namespace,
                "limit": limit,
                "enable_mqe": enable_mqe,
                "enable_hyde": enable_hyde,
                "enable_keyword_rerank": enable_keyword_rerank,
            }
        )
        chunks = list(self.rag_tool.last_retrieved_chunks)
        if not chunks:
            return generated_answer

        useful_chunks = self._select_useful_chunks(normalized, chunks)
        evidence = self._format_retrieved_chunks(useful_chunks, question=normalized)
        if not evidence:
            return generated_answer
        return "\n\n".join([generated_answer, "检索证据：", evidence])

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

    def rag_inventory(self) -> str:
        """Return document/chunk metadata for the current RAG namespace.

        向量库负责“相似度召回”，但它不适合直接展示文档来源、解析器和
        chunk 数。这里读取 document_store，把第八章 RAG 架构里的文档库状态
        暴露给页面，方便确认上传文件是否真的完成了入库和切块。
        """

        document_store = getattr(self.rag_tool, "document_store", None)
        if document_store is None or not hasattr(document_store, "list_documents"):
            return "未配置 RAG 文档库，无法查看文档清单。"

        documents = document_store.list_documents(namespace=self.rag_namespace)
        rows: list[dict[str, Any]] = []
        for document in documents:
            document_id = str(document.get("document_id", ""))
            chunks = (
                document_store.list_chunks(document_id, namespace=self.rag_namespace)
                if hasattr(document_store, "list_chunks")
                else []
            )
            rows.append(
                {
                    "document_id": document_id,
                    "namespace": document.get("namespace"),
                    "chunk_count": len(chunks),
                    "parser": document.get("parser"),
                    "source_path": document.get("source_path"),
                    "updated_at": document.get("updated_at"),
                    "metadata": document.get("metadata") or {},
                }
            )

        payload = {
            "namespace": self.rag_namespace,
            "storage_path": str(getattr(document_store, "path", "")),
            "document_count": len(rows),
            "documents": rows,
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def trace(self) -> str:
        return json.dumps(self._trace_events(), ensure_ascii=False, indent=2)

    def _trace_events(self) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for source in (self.perception_tool, self.rag_tool, self.memory_manager, self.assistant):
            events.extend(list(getattr(source, "trace_events", []) or []))
        return events

    def _format_retrieved_chunks(
        self,
        chunks: list[dict[str, Any]],
        *,
        question: str = "",
    ) -> str:
        lines: list[str] = []
        for index, chunk in enumerate(chunks, 1):
            content = self._clean_chunk_content(
                str(chunk.get("content", "")),
                question=question,
                max_chars=240,
            )
            lines.append(
                f"{index}. document={chunk.get('document_id')} "
                f"chunk={chunk.get('chunk_index')} "
                f"score={float(chunk.get('score', 0.0)):.3f} "
                f"vector={float(chunk.get('vector_score', chunk.get('score', 0.0))):.3f} "
                f"keyword={float(chunk.get('keyword_score', 0.0)):.3f}\n"
                f"   {content}"
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
        fallback_hits = [
            chunk
            for chunk in chunks
            if self._question_overlap_score(question, str(chunk.get("content", ""))) >= 3
        ]
        candidates = keyword_hits or positive or fallback_hits
        return candidates[:3]

    def _question_overlap_score(self, question: str, content: str) -> int:
        """Score a chunk when the retriever gives no positive relevance signal."""

        ignored_chars = set("的是了么吗什么哪些可以包括一个当前当前和与及在中为有")
        question_chars = {
            char
            for char in question
            if "\u4e00" <= char <= "\u9fff" and char not in ignored_chars
        }
        content_chars = {
            char
            for char in content
            if "\u4e00" <= char <= "\u9fff" and char not in ignored_chars
        }
        return len(question_chars & content_chars)

    def _clean_chunk_content(
        self,
        content: str,
        *,
        question: str = "",
        max_chars: int = 360,
    ) -> str:
        raw = content.strip()
        if raw.startswith("# Perceptual source:") and "\n\n" in raw:
            # Perceptual RAG chunks prepend source metadata before extracted text.
            raw = raw.split("\n\n", 1)[1].strip()
        raw = self._trim_to_question_focus(raw, question)
        normalized = " ".join(raw.split())
        if len(normalized) <= max_chars:
            return normalized
        return f"{normalized[:max_chars].rstrip()}..."

    def _trim_to_question_focus(self, content: str, question: str) -> str:
        """Trim a chunk to the subsection that directly answers field/list asks."""

        normalized_question = question.strip()
        if not normalized_question:
            return content
        if not any(marker in normalized_question for marker in ("哪些", "包括", "字段")):
            return content

        anchors = [
            "统一交互协议可包括",
            "统一协议字段",
            "协议字段",
            "可包括",
        ]
        for anchor in anchors:
            index = content.find(anchor)
            if index >= 0:
                return content[index:].lstrip("# \n")
        return content


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
            rag_inventory_output = gr.Code(label="RAG 文档库", language="json")
            gr.Button("检索记忆").click(dashboard.recall, inputs=recall_query, outputs=recall_output)
            gr.Button("刷新库存").click(dashboard.memory_inventory, outputs=inventory_output)
            gr.Button("刷新 RAG 文档库").click(dashboard.rag_inventory, outputs=rag_inventory_output)

        with gr.Tab("Trace"):
            trace_output = gr.Code(label="Trace", language="json")
            gr.Button("刷新 Trace").click(dashboard.trace, outputs=trace_output)

    return demo
