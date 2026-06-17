from __future__ import annotations

import json

from app.note_tool import NoteTool
from app.context_engineering import ContextBuilder, ContextConfig, ContextPacket


def test_note_tool_adds_searches_updates_and_removes_notes(tmp_path):
    tool = NoteTool(path=tmp_path / "notes.jsonl")

    add_result = tool.run(
        {
            "action": "add",
            "content": "第九章上下文工程采用 GSSC 流程。",
            "title": "上下文工程",
            "tags": ["chapter9", "context"],
            "importance": 0.9,
        }
    )

    assert "已保存笔记" in add_result
    note_id = tool.notes[0]["note_id"]
    assert tool.validate_parameters({"action": "search", "query": "GSSC"})

    search_result = tool.run({"action": "search", "query": "GSSC", "limit": 3})
    assert "上下文工程采用 GSSC" in search_result
    assert "chapter9" in search_result

    update_result = tool.run(
        {
            "action": "update",
            "note_id": note_id,
            "content": "GSSC 包括 Gather、Select、Structure、Compress。",
            "tags": "chapter9,gssc",
        }
    )
    assert "已更新笔记" in update_result
    assert "Compress" in tool.notes[0]["content"]
    assert tool.notes[0]["tags"] == ["chapter9", "gssc"]

    remove_result = tool.run({"action": "remove", "note_id": note_id})
    assert "已删除笔记" in remove_result
    assert tool.notes == []


def test_note_tool_persists_notes_as_jsonl(tmp_path):
    path = tmp_path / "notes.jsonl"
    tool = NoteTool(path=path)

    tool.run({"action": "add", "content": "需要把 RAG 证据放入 Evidence 区块。"})

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["content"] == "需要把 RAG 证据放入 Evidence 区块。"

    reloaded = NoteTool(path=path)
    assert len(reloaded.notes) == 1
    assert "Evidence 区块" in reloaded.run({"action": "list"})


def test_note_tool_exposes_hello_agents_tool_schema(tmp_path):
    tool = NoteTool(path=tmp_path / "notes.jsonl")
    names = {parameter.name for parameter in tool.get_parameters()}

    assert tool.name == "note"
    assert {"action", "content", "query", "note_id", "tags", "limit"}.issubset(names)
    assert tool.validate_parameters({"action": "add", "content": "记录一条笔记"})
    assert tool.validate_parameters({"action": "list"})
    assert tool.validate_parameters({"action": "stats"})
    assert not tool.validate_parameters({"action": "add"})


def test_note_tool_search_result_can_feed_context_builder(tmp_path):
    note_tool = NoteTool(path=tmp_path / "notes.jsonl")
    note_tool.run(
        {
            "action": "add",
            "title": "上下文结构",
            "content": "上下文应包含 Role、Task、Evidence、Context、Output 五段式结构。",
            "tags": ["chapter9", "context"],
            "importance": 0.9,
        }
    )
    note_text = note_tool.run({"action": "search", "query": "Evidence Context Output"})

    builder = ContextBuilder(config=ContextConfig(max_tokens=300))
    result = builder.build(
        user_query="上下文工程的 prompt 应该怎么组织？",
        system_instructions="你是上下文工程教学助手。",
        custom_packets=[
            ContextPacket(
                content=note_text,
                source="note",
                relevance_score=0.85,
                metadata={"section": "context", "type": "task_note"},
            )
        ],
        output_instructions="基于笔记简洁回答。",
    )

    assert "[Context]" in result.context
    assert "source=note" in result.context
    assert "Evidence、Context、Output" in result.context
    assert any(event["stage"] == "context.select" for event in result.trace)
