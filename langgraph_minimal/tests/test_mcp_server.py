from __future__ import annotations

import anyio

from app.mcp_server import create_mcp_server


def test_mcp_server_exposes_terminal_and_note_tools(tmp_path):
    async def scenario():
        (tmp_path / "README.md").write_text("MCP server can expose local tools.\n", encoding="utf-8")
        server = create_mcp_server(
            root=tmp_path,
            note_path=tmp_path / "notes.jsonl",
            include_agent_tools=False,
        )

        tools = await server.list_tools()
        names = {tool.name for tool in tools}

        assert {"terminal_run", "note_run"}.issubset(names)

    anyio.run(scenario)


def test_mcp_server_calls_terminal_tool(tmp_path):
    async def scenario():
        (tmp_path / "README.md").write_text("TerminalTool over MCP.\n", encoding="utf-8")
        server = create_mcp_server(
            root=tmp_path,
            note_path=tmp_path / "notes.jsonl",
            include_agent_tools=False,
        )

        result = await server.call_tool(
            "terminal_run",
            {"action": "read", "path": "README.md"},
        )

        assert "TerminalTool over MCP" in _mcp_text(result)

    anyio.run(scenario)


def test_mcp_server_calls_note_tool_with_persistence(tmp_path):
    async def scenario():
        note_path = tmp_path / "notes.jsonl"
        server = create_mcp_server(
            root=tmp_path,
            note_path=note_path,
            include_agent_tools=False,
        )

        add_result = await server.call_tool(
            "note_run",
            {
                "action": "add",
                "title": "MCP note",
                "content": "MCP 可以把 NoteTool 暴露给外部 Agent。",
                "tags": ["chapter10", "mcp"],
            },
        )
        search_result = await server.call_tool(
            "note_run",
            {"action": "search", "query": "NoteTool MCP", "limit": 3},
        )

        assert "已保存笔记" in _mcp_text(add_result)
        assert "NoteTool 暴露给外部 Agent" in _mcp_text(search_result)
        assert "chapter10" in note_path.read_text(encoding="utf-8")

    anyio.run(scenario)


def _mcp_text(result) -> str:
    if isinstance(result, dict):
        return str(result)
    if isinstance(result, tuple) and result:
        return _mcp_text(result[0])
    parts = []
    for item in result:
        parts.append(str(getattr(item, "text", item)))
    return "\n".join(parts)
