from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.context_engineering import ContextBuilder, ContextBuildResult, ContextConfig, ContextPacket
from app.my_memory_system import MyMemoryTool
from app.note_tool import NoteTool
from app.terminal_tool import TerminalTool


@dataclass
class CodebaseExploreResult:
    """Structured result for one repository exploration pass."""

    structure: str
    todos: str
    trace: list[dict[str, Any]]


class CodebaseMaintenanceAssistant:
    """Long-running codebase maintenance assistant.

    它不是直接调用 LLM 的“聊天壳”，而是一个上下文工程编排器：
    - TerminalTool 负责读取代码库现场证据；
    - NoteTool 负责记录问题、改进点和长期重构任务；
    - MemoryTool 负责保存本轮维护状态，帮助跨轮保持连贯；
    - ContextBuilder 在 token 预算下把这些材料组织成可交给 LLM 的 prompt。
    """

    def __init__(
        self,
        *,
        root: str | Path,
        note_path: str | Path | None = None,
        user_id: str = "codebase_maintainer",
        context_max_tokens: int = 2000,
        terminal_max_bytes: int = 12_000,
        memory_tool: MyMemoryTool | None = None,
        note_tool: NoteTool | None = None,
        terminal_tool: TerminalTool | None = None,
        llm: Any | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.terminal_tool = terminal_tool or TerminalTool(
            root=self.root,
            max_bytes=terminal_max_bytes,
        )
        self.note_tool = note_tool or NoteTool(
            path=Path(note_path or self.root / ".maintenance_notes.jsonl"),
            user_id=user_id,
        )
        self.memory_tool = memory_tool or MyMemoryTool(user_id=user_id)
        self.llm = llm
        self.context_config = ContextConfig(max_tokens=context_max_tokens)
        self.conversation_history: list[dict[str, Any]] = []
        self.trace_events: list[dict[str, Any]] = []

    def run(self, user_input: str, *, mode: str = "auto", **llm_kwargs: Any) -> str:
        """Run one long-horizon maintenance turn.

        mode 借鉴第 9.6 节的长程智能体设计：同一个入口根据任务意图
        触发不同预处理，但最终都回到 ContextBuilder 统一构建上下文。
        """

        normalized_mode = self._normalize_mode(mode)
        self._preprocess_by_mode(user_input, normalized_mode)
        context_result = self.build_maintenance_context(
            self._mode_question(user_input, normalized_mode)
        )
        messages = [{"role": "user", "content": context_result.context}]
        response = self._invoke_llm(messages, **llm_kwargs)
        response = self._postprocess_response(
            user_input=user_input,
            mode=normalized_mode,
            response=response,
            context_result=context_result,
        )
        self.conversation_history.extend(
            [
                {"role": "user", "content": user_input, "mode": normalized_mode},
                {"role": "assistant", "content": response, "mode": normalized_mode},
            ]
        )
        self.trace_events.append(
            {
                "stage": "maintenance.run",
                "mode": normalized_mode,
                "input": user_input,
                "history_messages": len(self.conversation_history),
                "context_tokens": context_result.total_tokens,
            }
        )
        return response

    def explore(self, paths: list[str] | None = None) -> CodebaseExploreResult:
        """Explore visible repository structure and TODO-like maintenance signals."""

        paths = paths or ["."]
        structure_blocks: list[str] = []
        for path in paths:
            structure_blocks.append(
                self.terminal_tool.run(
                    {
                        "action": "list",
                        "path": path,
                        "limit": 40,
                    }
                )
            )

        todos = self.terminal_tool.run(
            {
                "action": "search",
                "path": ".",
                "query": "TODO",
                "pattern": "*.py",
                "limit": 20,
            }
        )
        structure = "\n\n".join(structure_blocks)
        self.memory_tool.run(
            {
                "action": "add",
                "memory_type": "working",
                "content": f"代码库探索完成，已查看路径：{', '.join(paths)}。",
                "importance": 0.6,
            }
        )
        self.trace_events.append(
            {
                "stage": "maintenance.explore",
                "paths": paths,
                "structure_blocks": len(structure_blocks),
            }
        )
        return CodebaseExploreResult(
            structure=structure,
            todos=todos,
            trace=self._combined_trace(),
        )

    def record_issue(
        self,
        *,
        title: str,
        content: str,
        path: str = "",
        importance: float = 0.7,
    ) -> str:
        """Persist a discovered code smell, bug risk, or improvement point."""

        result = self.note_tool.run(
            {
                "action": "add",
                "title": title,
                "content": content,
                "tags": ["codebase", "issue", path] if path else ["codebase", "issue"],
                "importance": importance,
            }
        )
        self.memory_tool.run(
            {
                "action": "add",
                "memory_type": "working",
                "content": f"发现维护问题：{title} {path}".strip(),
                "importance": min(1.0, importance),
            }
        )
        self.trace_events.append(
            {"stage": "maintenance.record_issue", "title": title, "path": path}
        )
        return result

    def track_refactor_task(
        self,
        *,
        title: str,
        content: str,
        path: str = "",
        status: str = "todo",
        importance: float = 0.8,
    ) -> str:
        """Persist a long-running refactor task as a searchable task note."""

        task_content = f"status={status}\npath={path or '-'}\n{content}"
        result = self.note_tool.run(
            {
                "action": "add",
                "title": title,
                "content": task_content,
                "tags": ["codebase", "refactor", status, path] if path else ["codebase", "refactor", status],
                "importance": importance,
            }
        )
        self.memory_tool.run(
            {
                "action": "add",
                "memory_type": "working",
                "content": f"长期重构任务已记录：{title}，状态 {status}。",
                "importance": min(1.0, importance),
            }
        )
        self.trace_events.append(
            {
                "stage": "maintenance.track_refactor",
                "title": title,
                "path": path,
                "status": status,
            }
        )
        return result

    def analyze(self, target: str) -> str:
        """Convenience wrapper for code analysis mode."""

        return self.run(f"分析 {target} 的维护风险", mode="analyze")

    def plan_next_steps(self, topic: str = "当前代码库") -> str:
        """Convenience wrapper for planning mode."""

        return self.run(f"规划 {topic} 的下一步维护工作", mode="plan")

    def create_note(
        self,
        *,
        title: str,
        content: str,
        tags: list[str] | None = None,
        importance: float = 0.7,
    ) -> str:
        """Expose an explicit note-taking path for human/agent collaboration."""

        return self.note_tool.run(
            {
                "action": "add",
                "title": title,
                "content": content,
                "tags": tags or ["codebase", "manual"],
                "importance": importance,
            }
        )

    def build_maintenance_context(self, question: str) -> ContextBuildResult:
        """Assemble a bounded context for the next maintenance step."""

        structure = self.terminal_tool.run({"action": "list", "path": ".", "limit": 40})
        todos = self.terminal_tool.run(
            {
                "action": "search",
                "path": ".",
                "query": "TODO",
                "pattern": "*.py",
                "limit": 20,
            }
        )
        related_notes = self.note_tool.run(
            {
                "action": "search",
                "query": f"{question} codebase refactor issue TODO",
                "limit": 6,
            }
        )
        related_memory = self.memory_tool.run(
            {
                "action": "search",
                "memory_type": "all",
                "query": question,
                "limit": 5,
            }
        )

        builder = ContextBuilder(config=self.context_config)
        result = builder.build(
            user_query=question,
            system_instructions=(
                "你是代码库维护助手。回答时必须区分现场证据、历史笔记和长期任务；"
                "不要编造没有出现在上下文中的文件或结论。"
            ),
            custom_packets=[
                ContextPacket(
                    content=f"{structure}\n\n{todos}",
                    source="terminal",
                    relevance_score=0.9,
                    metadata={"section": "evidence", "type": "codebase_snapshot"},
                ),
                ContextPacket(
                    content=related_notes,
                    source="note",
                    relevance_score=0.85,
                    metadata={"section": "context", "type": "maintenance_notes"},
                ),
                ContextPacket(
                    content=related_memory,
                    source="memory",
                    relevance_score=0.8,
                    metadata={"section": "context", "type": "maintenance_memory"},
                ),
            ],
            output_instructions=(
                "输出建议包含：当前理解、发现的问题、下一步维护计划、需要继续追踪的任务。"
            ),
        )
        result.trace.append(
            {
                "stage": "maintenance.context",
                "question": question,
                "selected_sources": [packet.source for packet in result.packets],
            }
        )
        self.trace_events.append(result.trace[-1])
        return result

    def get_stats(self) -> dict[str, Any]:
        """Return observable long-running assistant state."""

        stats = {
            "root": str(self.root),
            "notes": len(self.note_tool.notes),
            "history_messages": len(self.conversation_history),
            "trace_events": len(self.trace_events),
            "terminal_events": len(self.terminal_tool.trace_events),
            "note_events": len(self.note_tool.trace_events),
            "memory_events": len(self.memory_tool.trace_events),
        }
        self.trace_events.append({"stage": "maintenance.stats", **stats})
        return stats

    def generate_report(self) -> str:
        """Generate a deterministic report from persisted maintenance notes."""

        notes = self.note_tool.run({"action": "list", "limit": 20})
        stats = self.get_stats()
        self.trace_events.append(
            {"stage": "maintenance.report", "notes": stats["notes"]}
        )
        return (
            "# 代码库维护报告\n\n"
            f"- root: {stats['root']}\n"
            f"- notes: {stats['notes']}\n"
            f"- history_messages: {stats['history_messages']}\n\n"
            "## 维护笔记\n"
            f"{notes}"
        )

    def _normalize_mode(self, mode: str) -> str:
        normalized = str(mode or "auto").strip().lower()
        if normalized not in {"auto", "explore", "analyze", "plan", "refactor"}:
            return "auto"
        return normalized

    def _preprocess_by_mode(self, user_input: str, mode: str) -> None:
        if mode == "explore":
            self.explore()
        elif mode == "analyze":
            self.terminal_tool.run(
                {
                    "action": "search",
                    "path": ".",
                    "query": "TODO",
                    "pattern": "*.py",
                    "limit": 20,
                }
            )
        elif mode in {"plan", "refactor"}:
            self.note_tool.run(
                {
                    "action": "search",
                    "query": f"{user_input} refactor todo codebase",
                    "limit": 6,
                }
            )
        self.trace_events.append(
            {"stage": "maintenance.preprocess", "mode": mode, "input": user_input}
        )

    def _mode_question(self, user_input: str, mode: str) -> str:
        if mode == "explore":
            return f"请基于当前代码库结构回答：{user_input}"
        if mode == "analyze":
            return f"请分析代码维护风险并给出证据：{user_input}"
        if mode == "plan":
            return f"请结合长期笔记制定下一步计划：{user_input}"
        if mode == "refactor":
            return f"请结合长期任务推进重构：{user_input}"
        return user_input

    def _invoke_llm(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        if self.llm is None:
            return (
                "已构建代码库维护上下文。请基于 Evidence 查看现场代码，"
                "基于 Context 延续历史笔记和长期重构任务。"
            )
        if hasattr(self.llm, "invoke"):
            return str(self.llm.invoke(messages, **kwargs))
        if callable(self.llm):
            return str(self.llm(messages, **kwargs))
        raise TypeError("llm 必须提供 invoke(messages, **kwargs) 或可调用接口")

    def _postprocess_response(
        self,
        *,
        user_input: str,
        mode: str,
        response: str,
        context_result: ContextBuildResult,
    ) -> str:
        self.note_tool.run(
            {
                "action": "add",
                "title": f"{mode}: {user_input}",
                "content": (
                    f"mode={mode}\n"
                    f"context_tokens={context_result.total_tokens}\n"
                    f"assistant_response={response}"
                ),
                "tags": ["codebase", "run", mode],
                "importance": 0.6,
            }
        )
        self.memory_tool.run(
            {
                "action": "add",
                "memory_type": "working",
                "content": f"完成一次 {mode} 模式维护问答：{user_input}",
                "importance": 0.6,
            }
        )
        self.trace_events.append(
            {
                "stage": "maintenance.postprocess",
                "mode": mode,
                "context_tokens": context_result.total_tokens,
            }
        )
        return response

    def _combined_trace(self) -> list[dict[str, Any]]:
        return (
            list(self.trace_events)
            + list(self.terminal_tool.trace_events)
            + list(self.note_tool.trace_events)
            + list(self.memory_tool.trace_events)
        )
