from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .manager import MyMemoryManager
from .rag_qa_demo import RAGQAResult
from .rag_tool import MyRAGTool


@dataclass
class DocumentLoadResult:
    """Result returned after a learning document is loaded."""

    document_id: str
    raw_output: str
    trace: list[dict[str, Any]]


@dataclass
class DocumentLearningAnswer:
    """Answer plus the retrieval/memory evidence used by the assistant."""

    answer: str
    references: list[str]
    memory_context: str
    trace: list[dict[str, Any]]
    raw_output: str


class DocumentLearningAssistant:
    """Chapter 8 style assistant that combines RAG and memory.

    第八章的实战案例不只是“问知识库”，还要让 Agent 记住学习过程：
    - 文档进入 RAG 知识库；
    - 学习事件进入情景记忆；
    - 当前问题进入工作记忆；
    - 回答时同时参考知识库和已有学习记忆。
    """

    def __init__(
        self,
        *,
        rag_tool: MyRAGTool | None = None,
        memory_manager: MyMemoryManager | None = None,
        namespace: str = "chapter8_learning",
        session_id: str = "default_session",
    ) -> None:
        self.rag_tool = rag_tool or MyRAGTool()
        self.memory_manager = memory_manager or MyMemoryManager(user_id="learner")
        self.namespace = namespace
        self.session_id = session_id
        self.trace_events: list[dict[str, Any]] = []

    def load_document(
        self,
        file_path: str | Path,
        *,
        document_id: str | None = None,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
    ) -> DocumentLoadResult:
        path = Path(file_path)
        effective_document_id = document_id or path.name
        output = self.rag_tool.run(
            {
                "action": "add_document",
                "file_path": str(path),
                "document_id": effective_document_id,
                "namespace": self.namespace,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
            }
        )

        # 文档加载是一个发生过的学习事件，因此写入情景记忆，后续可生成学习报告。
        self.memory_manager.add(
            content=f"加载学习文档：{effective_document_id}",
            memory_type="episodic",
            importance=0.7,
            metadata={
                "session_id": self.session_id,
                "event_type": "document_loaded",
                "document_id": effective_document_id,
                "file_path": str(path),
            },
        )

        self._append_trace(
            {
                "stage": "learning.load_document",
                "document_id": effective_document_id,
                "namespace": self.namespace,
            }
        )
        return DocumentLoadResult(
            document_id=effective_document_id,
            raw_output=output,
            trace=list(self.trace_events),
        )

    def ask(
        self,
        question: str,
        *,
        limit: int = 5,
        memory_limit: int = 3,
        enable_mqe: bool = True,
        enable_hyde: bool = False,
    ) -> DocumentLearningAnswer:
        normalized_question = question.strip()
        memory_hits = self.memory_manager.search(
            query=normalized_question,
            memory_type="all",
            limit=memory_limit,
            session_id=self.session_id,
        )
        memory_context = self._format_memory_context(memory_hits)

        rag_output = self.rag_tool.run(
            {
                "action": "ask",
                "question": normalized_question,
                "namespace": self.namespace,
                "limit": limit,
                "enable_mqe": enable_mqe,
                "enable_hyde": enable_hyde,
            }
        )
        rag_result = RAGQAResult(
            answer=self._extract_answer(rag_output),
            references=self._extract_references(rag_output),
            trace=list(self.rag_tool.trace_events),
            raw_output=rag_output,
        )

        # 工作记忆保留当前任务焦点，情景记忆记录一次完整学习行为。
        self.memory_manager.add(
            content=f"当前学习问题：{normalized_question}",
            memory_type="working",
            importance=0.6,
            metadata={"session_id": self.session_id, "event_type": "current_question"},
        )
        self.memory_manager.add(
            content=f"完成一次文档问答：{normalized_question}",
            memory_type="episodic",
            importance=0.75,
            metadata={"session_id": self.session_id, "event_type": "document_qa"},
        )

        self._append_trace(
            {
                "stage": "learning.ask",
                "question": normalized_question,
                "memory_hits": len(memory_hits),
                "references": len(rag_result.references),
            }
        )
        return DocumentLearningAnswer(
            answer=rag_result.answer,
            references=rag_result.references,
            memory_context=memory_context,
            trace=list(self.trace_events),
            raw_output=rag_output,
        )

    def learning_report(self, *, limit: int = 10) -> str:
        """Build a lightweight review report from session memories."""

        episodic_records = self.memory_manager.summary(
            memory_type="episodic",
            limit=limit,
            session_id=self.session_id,
        )
        working_records = self.memory_manager.summary(
            memory_type="working",
            limit=limit,
            session_id=self.session_id,
        )

        lines = ["学习报告", "一、学习事件"]
        if episodic_records:
            lines.extend(f"- {record.content}" for record in episodic_records)
        else:
            lines.append("- 暂无学习事件")

        lines.append("二、当前关注点")
        if working_records:
            lines.extend(f"- {record.content}" for record in working_records)
        else:
            lines.append("- 暂无当前关注点")

        self._append_trace(
            {
                "stage": "learning.report",
                "episodic_count": len(episodic_records),
                "working_count": len(working_records),
            }
        )
        return "\n".join(lines)

    def _append_trace(self, event: dict[str, Any]) -> None:
        # 每次对外返回 trace 时，把 RAG 与 Memory 的最新事件合并，方便调试认知闭环。
        self.trace_events = [
            *self.rag_tool.trace_events,
            *self.memory_manager.trace_events,
            event,
        ]

    def _format_memory_context(self, hits: list[Any]) -> str:
        if not hits:
            return ""
        lines = ["相关学习记忆："]
        for index, item in enumerate(hits, 1):
            lines.append(
                f"{index}. [{item.record.memory_type}] "
                f"score={item.score:.2f} {item.record.content}"
            )
        return "\n".join(lines)

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
