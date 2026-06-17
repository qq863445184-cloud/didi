from __future__ import annotations

from app.context_engineering import ContextBuilder, ContextConfig, ContextPacket
from app.terminal_tool import TerminalTool


def test_terminal_tool_lists_reads_and_searches_files(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    readme = docs / "README.md"
    readme.write_text(
        "# Chapter 9\n\nTerminalTool gives agents read-only filesystem context.\n",
        encoding="utf-8",
    )

    tool = TerminalTool(root=tmp_path)

    list_result = tool.run({"action": "list", "path": "docs"})
    assert "README.md" in list_result
    assert "type=file" in list_result

    read_result = tool.run({"action": "read", "path": "docs/README.md"})
    assert "TerminalTool gives agents" in read_result
    assert "bytes=" in read_result

    search_result = tool.run({"action": "search", "query": "read-only", "pattern": "*.md"})
    assert "docs/README.md:3" in search_result
    assert "read-only filesystem context" in search_result
    assert any(event["stage"] == "terminal.search" for event in tool.trace_events)


def test_terminal_tool_blocks_path_escape(tmp_path):
    outside = tmp_path.parent / "outside_secret.txt"
    outside.write_text("secret", encoding="utf-8")
    tool = TerminalTool(root=tmp_path)

    result = tool.run({"action": "read", "path": "../outside_secret.txt"})

    assert "错误" in result
    assert "访问被拒绝" in result
    assert "secret" not in result
    assert tool.trace_events[-1]["stage"] == "terminal.error"


def test_terminal_tool_truncates_large_reads(tmp_path):
    big = tmp_path / "big.txt"
    big.write_text("0123456789" * 20, encoding="utf-8")
    tool = TerminalTool(root=tmp_path, max_bytes=25)

    result = tool.run({"action": "read", "path": "big.txt"})

    assert "0123456789012345678901234" in result
    assert "[truncated" in result
    assert "bytes=200" in result


def test_terminal_tool_exposes_hello_agents_schema(tmp_path):
    tool = TerminalTool(root=tmp_path)
    names = {parameter.name for parameter in tool.get_parameters()}

    assert tool.name == "terminal"
    assert {"action", "path", "query", "pattern", "limit", "max_bytes"}.issubset(names)
    assert tool.validate_parameters({"action": "pwd"})
    assert tool.validate_parameters({"action": "list", "path": "."})
    assert tool.validate_parameters({"action": "read", "path": "README.md"})
    assert tool.validate_parameters({"action": "search", "query": "agent"})
    assert not tool.validate_parameters({"action": "search"})


def test_terminal_tool_result_can_feed_context_builder(tmp_path):
    note = tmp_path / "architecture.md"
    note.write_text(
        "TerminalTool should be used as evidence when the agent needs live file context.",
        encoding="utf-8",
    )
    tool = TerminalTool(root=tmp_path)
    file_text = tool.run({"action": "read", "path": "architecture.md"})

    builder = ContextBuilder(config=ContextConfig(max_tokens=240))
    result = builder.build(
        user_query="TerminalTool 应该放到上下文哪里？",
        system_instructions="你是上下文工程教学助手。",
        custom_packets=[
            ContextPacket(
                content=file_text,
                source="terminal",
                relevance_score=0.9,
                metadata={"section": "evidence", "type": "filesystem"},
            )
        ],
        output_instructions="基于文件证据回答。",
    )

    assert "[Evidence]" in result.context
    assert "source=terminal" in result.context
    assert "live file context" in result.context
