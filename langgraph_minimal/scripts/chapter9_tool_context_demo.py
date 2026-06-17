from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.context_engineering import ContextBuilder, ContextConfig, ContextPacket
from app.my_memory_system import MyMemoryTool
from app.note_tool import NoteTool
from app.terminal_tool import TerminalTool


def build_tool_context_demo(
    *,
    workspace_root: str | Path | None = None,
    note_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build one explainable context from TerminalTool, MemoryTool and NoteTool.

    这个 demo 展示的是“整理流程”：工具各自负责获取信息，但进入
    ContextBuilder 前都被统一包装成 ContextPacket。这样后续选择、
    分区、压缩、trace 都可以走同一套上下文工程管线。
    """

    root = Path(workspace_root or ROOT / "memory_data" / "chapter9_tool_context_demo")
    root.mkdir(parents=True, exist_ok=True)
    project_note = root / "project_context.md"
    project_note.write_text(
        "# 第九章工具协同\n\n"
        "TerminalTool 读取项目现场证据，适合放入 Evidence 区块。\n"
        "MemoryTool 检索用户长期或短期记忆，适合补充 Context。\n"
        "NoteTool 检索任务笔记和人工标注，也适合补充 Context。\n",
        encoding="utf-8",
    )

    terminal_tool = TerminalTool(root=root)
    memory_tool = MyMemoryTool(user_id="chapter9_demo")
    note_tool = NoteTool(
        path=Path(note_path or root / "chapter9_notes.jsonl"),
        user_id="chapter9_demo",
    )

    memory_tool.run(
        {
            "action": "add",
            "memory_type": "working",
            "content": "用户正在学习第九章上下文工程，关注工具如何协同构建 prompt。",
            "importance": 0.8,
        }
    )
    note_tool.run(
        {
            "action": "add",
            "title": "工具协同规则",
            "content": "工具输出统一包装为 ContextPacket，再交给 ContextBuilder 进行选择和结构化。",
            "tags": ["chapter9", "tool", "context"],
            "importance": 0.9,
        }
    )

    question = "TerminalTool、MemoryTool、NoteTool 怎么协同进入 ContextBuilder？"
    terminal_output = terminal_tool.run(
        {"action": "read", "path": "project_context.md", "max_bytes": 4000}
    )
    memory_output = memory_tool.run(
        {
            "action": "search",
            "memory_type": "all",
            "query": "第九章 上下文工程 工具 协同",
            "limit": 3,
        }
    )
    note_output = note_tool.run(
        {
            "action": "search",
            "query": "工具 ContextPacket ContextBuilder",
            "limit": 3,
        }
    )

    builder = ContextBuilder(config=ContextConfig(max_tokens=1000))
    build_result = builder.build(
        user_query=question,
        system_instructions="你是第九章上下文工程教学助手，必须解释证据来源和上下文分区。",
        custom_packets=[
            ContextPacket(
                content=terminal_output,
                source="terminal",
                relevance_score=0.95,
                metadata={"section": "evidence", "type": "filesystem"},
            ),
            ContextPacket(
                content=memory_output,
                source="memory",
                relevance_score=0.8,
                metadata={"section": "context", "type": "user_memory"},
            ),
            ContextPacket(
                content=note_output,
                source="note",
                relevance_score=0.85,
                metadata={"section": "context", "type": "task_note"},
            ),
        ],
        output_instructions="按 TerminalTool、MemoryTool、NoteTool、ContextBuilder 四部分说明。",
    )

    return {
        "question": question,
        "terminal_output": terminal_output,
        "memory_output": memory_output,
        "note_output": note_output,
        "context": build_result.context,
        "selected_sources": [packet.source for packet in build_result.packets],
        "terminal_trace": terminal_tool.trace_events,
        "memory_trace": memory_tool.trace_events,
        "note_trace": note_tool.trace_events,
        "context_trace": build_result.trace,
    }


def main() -> None:
    result = build_tool_context_demo()
    print("[question]")
    print(result["question"])
    print("\n[terminal output]")
    print(result["terminal_output"])
    print("\n[memory output]")
    print(result["memory_output"])
    print("\n[note output]")
    print(result["note_output"])
    print("\n[assembled context]")
    print(result["context"])
    print("\n[trace]")
    print(
        json.dumps(
            {
                "terminal": result["terminal_trace"],
                "memory": result["memory_trace"],
                "note": result["note_trace"],
                "context": result["context_trace"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
