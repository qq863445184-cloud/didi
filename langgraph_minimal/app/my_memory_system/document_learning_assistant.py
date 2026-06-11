from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Any

from .manager import MyMemoryManager
from .rag_qa_demo import RAGQAResult
from .rag_tool import MyRAGTool
from .stores import PerceptualMemoryStore, WorkingMemoryStore


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
    route: str = "hybrid"


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
        self.memory_manager = memory_manager or self._build_default_memory_manager()
        self.namespace = namespace
        self.session_id = session_id
        self.session_start = time()
        self.documents_loaded = 0
        self.questions_asked = 0
        self.concepts_learned = 0
        self.current_document: str | None = None
        self._learning_events: list[dict[str, Any]] = []
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
        self.documents_loaded += 1
        self.current_document = effective_document_id

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
        self.questions_asked += 1

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
            route="hybrid",
        )

    def route_query(self, question: str) -> str:
        """Choose a retrieval path for a learning question.

        这不是替代模型判断，而是教学版的透明路由器：文档证据问题走 RAG，
        学习历史/笔记问题走 Memory，两类线索都出现时走 Hybrid。
        """

        text = question.lower()
        rag_markers = {"文档", "资料", "根据", "原文", "知识库", "rag"}
        memory_markers = {"之前", "历史", "记得", "记忆", "笔记", "我保存", "学过"}
        wants_rag = any(marker in text for marker in rag_markers)
        wants_memory = any(marker in text for marker in memory_markers)

        if wants_rag and wants_memory:
            route = "hybrid"
        elif wants_memory:
            route = "memory"
        elif wants_rag or self.current_document:
            route = "rag"
        else:
            route = "hybrid"

        self._append_trace(
            {
                "stage": "learning.route",
                "question": question,
                "route": route,
                "wants_rag": wants_rag,
                "wants_memory": wants_memory,
            }
        )
        return route

    def ask_auto(self, question: str, *, limit: int = 5) -> DocumentLearningAnswer:
        """Answer by first routing to RAG, Memory, or Hybrid retrieval."""

        route = self.route_query(question)
        if route == "memory":
            recall_output = self.recall(question, limit=limit)
            return DocumentLearningAnswer(
                answer=recall_output,
                references=[],
                memory_context=recall_output,
                trace=list(self.trace_events),
                raw_output=recall_output,
                route=route,
            )

        answer = self.ask(question, limit=limit)
        answer.route = route
        self._append_trace(
            {
                "stage": "learning.ask_auto",
                "question": question,
                "route": route,
            }
        )
        answer.trace = list(self.trace_events)
        return answer

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
        semantic_records = self.memory_manager.summary(
            memory_type="semantic",
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

        lines.append("三、学习笔记")
        if semantic_records:
            lines.extend(f"- {record.content}" for record in semantic_records)
        else:
            lines.append("- 暂无学习笔记")

        self._append_trace(
            {
                "stage": "learning.report",
                "episodic_count": len(episodic_records),
                "working_count": len(working_records),
                "semantic_count": len(semantic_records),
            }
        )
        return "\n".join(lines)

    def add_note(self, note: str, *, importance: float = 0.7) -> str:
        """Save a learner's distilled note into semantic memory."""

        normalized_note = note.strip()
        semantic_record = self.memory_manager.add(
            content=normalized_note,
            memory_type="semantic",
            importance=importance,
            metadata={"session_id": self.session_id, "event_type": "learning_note"},
        )
        # 同步写一条工作记忆，让当前会话能立刻围绕这条笔记继续追问。
        self.memory_manager.add(
            content=f"刚添加学习笔记：{normalized_note}",
            memory_type="working",
            importance=min(1.0, importance),
            metadata={
                "session_id": self.session_id,
                "event_type": "current_note",
                "semantic_memory_id": semantic_record.memory_id,
            },
        )
        self.concepts_learned += 1
        self._append_trace(
            {
                "stage": "learning.add_note",
                "memory_id": semantic_record.memory_id,
                "importance": importance,
            }
        )
        return f"已保存学习笔记：{normalized_note}"

    def recall(
        self,
        query: str,
        *,
        memory_types: list[str] | None = None,
        limit: int = 5,
    ) -> str:
        """Recall notes and recent learning context from memory."""

        results = self.memory_manager.search(
            query=query.strip(),
            memory_type="all",
            memory_types=memory_types,
            limit=limit,
            session_id=self.session_id,
        )
        self._append_trace(
            {
                "stage": "learning.recall",
                "query": query,
                "hits": len(results),
                "memory_types": memory_types or "all",
            }
        )
        if not results:
            return "未找到相关学习记忆。"
        return self._format_memory_context(results)

    def get_stats(self) -> dict[str, Any]:
        """Expose store inventory for the learning assistant."""

        stats = self.memory_manager.stats()
        stats["learning_metrics"] = self._learning_metrics()
        self._append_trace(
            {
                "stage": "learning.stats",
                "total": stats["total"],
                **stats["learning_metrics"],
            }
        )
        return stats

    def generate_report(
        self,
        *,
        limit: int = 10,
        save_to_file: bool = False,
        file_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Build a structured report matching the chapter 8 example naming."""

        report = {
            "title": "学习报告",
            "session_id": self.session_id,
            "namespace": self.namespace,
            "learning_metrics": self._learning_metrics(),
            "memory_summary": {
                "episodic": [
                    record.content
                    for record in self.memory_manager.summary(
                        memory_type="episodic",
                        limit=limit,
                        session_id=self.session_id,
                    )
                ],
                "working": [
                    record.content
                    for record in self.memory_manager.summary(
                        memory_type="working",
                        limit=limit,
                        session_id=self.session_id,
                    )
                ],
                "semantic": [
                    record.content
                    for record in self.memory_manager.summary(
                        memory_type="semantic",
                        limit=limit,
                        session_id=self.session_id,
                    )
                ],
            },
            "rag_status": self.rag_tool.run({"action": "stats"}),
            "report_file": None,
        }
        if save_to_file:
            target = Path(file_path) if file_path is not None else Path("learning_report.json")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                json.dumps(report, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            report["report_file"] = str(target)

        self._append_trace(
            {
                "stage": "learning.generate_report",
                "save_to_file": save_to_file,
                "report_file": report["report_file"],
            }
        )
        return report

    def _append_trace(self, event: dict[str, Any]) -> None:
        self._learning_events.append(event)
        # 每次对外返回 trace 时，把 RAG 与 Memory 的最新事件合并，方便调试认知闭环。
        self.trace_events = [
            *self.rag_tool.trace_events,
            *self.memory_manager.trace_events,
            *self._learning_events,
        ]

    def _build_default_memory_manager(self) -> MyMemoryManager:
        """Create a no-service default memory stack for local demos.

        真正接 Qdrant/Neo4j 时可以从外部注入 manager；默认构造要能在教程
        smoke demo 里直接跑，所以 semantic/episodic 先用本地 store 承载。
        """

        return MyMemoryManager(
            user_id="learner",
            stores={
                "working": WorkingMemoryStore(),
                "semantic": WorkingMemoryStore(),
                "episodic": WorkingMemoryStore(),
                "perceptual": PerceptualMemoryStore(),
            },
        )

    def _learning_metrics(self) -> dict[str, Any]:
        return {
            "session_duration_seconds": round(time() - self.session_start, 3),
            "documents_loaded": self.documents_loaded,
            "questions_asked": self.questions_asked,
            "concepts_learned": self.concepts_learned,
            "current_document": self.current_document,
        }

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
