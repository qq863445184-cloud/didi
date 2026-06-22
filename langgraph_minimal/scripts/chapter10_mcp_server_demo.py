from __future__ import annotations

import json
import sys
from pathlib import Path

import anyio

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.mcp_server import create_mcp_server


async def main_async() -> None:
    demo_root = ROOT / "memory_data" / "chapter10_mcp_server_demo"
    demo_root.mkdir(parents=True, exist_ok=True)
    (demo_root / "README.md").write_text(
        "# Chapter 10 MCP Demo\n\nMCP exposes TerminalTool and NoteTool to external agents.\n",
        encoding="utf-8",
    )
    note_path = demo_root / "mcp_notes.jsonl"
    note_path.unlink(missing_ok=True)

    server = create_mcp_server(
        root=demo_root,
        note_path=note_path,
        user_id="chapter10_demo",
        include_agent_tools=False,
    )

    tools = await server.list_tools()
    print("[list_tools]")
    print(json.dumps([tool.name for tool in tools], ensure_ascii=False, indent=2))

    print("\n[terminal_run/read]")
    terminal_result = await server.call_tool(
        "terminal_run",
        {"action": "read", "path": "README.md"},
    )
    print(_mcp_text(terminal_result))

    print("\n[note_run/add]")
    add_result = await server.call_tool(
        "note_run",
        {
            "action": "add",
            "title": "MCP 工具协议",
            "content": "第十章通过 MCP 暴露 TerminalTool 和 NoteTool。",
            "tags": ["chapter10", "mcp"],
            "importance": 0.9,
        },
    )
    print(_mcp_text(add_result))

    print("\n[note_run/search]")
    search_result = await server.call_tool(
        "note_run",
        {"action": "search", "query": "TerminalTool NoteTool MCP", "limit": 3},
    )
    print(_mcp_text(search_result))


def _mcp_text(result) -> str:
    if isinstance(result, dict):
        return str(result)
    if isinstance(result, tuple) and result:
        return _mcp_text(result[0])
    return "\n".join(str(getattr(item, "text", item)) for item in result)


def main() -> None:
    anyio.run(main_async)


if __name__ == "__main__":
    main()
