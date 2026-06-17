from __future__ import annotations

from scripts.chapter9_tool_context_demo import build_tool_context_demo


def test_tool_context_demo_combines_terminal_memory_note_and_context(tmp_path):
    result = build_tool_context_demo(workspace_root=tmp_path, note_path=tmp_path / "notes.jsonl")

    assert "TerminalTool 读取项目现场证据" in result["terminal_output"]
    assert "用户正在学习第九章上下文工程" in result["memory_output"]
    assert "工具输出统一包装为 ContextPacket" in result["note_output"]

    context = result["context"]
    assert "[Evidence]" in context
    assert "[Context]" in context
    assert "source=terminal" in context
    assert "source=memory" in context
    assert "source=note" in context

    trace_stages = {
        event["stage"]
        for group_name in ["terminal_trace", "memory_trace", "note_trace", "context_trace"]
        for event in result[group_name]
    }
    assert {"terminal.read", "manager.search", "note.search", "context.build"}.issubset(trace_stages)
