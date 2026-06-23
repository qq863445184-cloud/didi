from __future__ import annotations

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hello_agents.tools import MCPTool


def main() -> None:
    demo_root = ROOT / "memory_data" / "chapter10_external_filesystem_mcp_demo"
    demo_root.mkdir(parents=True, exist_ok=True)
    readme = demo_root / "README.md"
    readme.write_text(
        "# External filesystem MCP\n\n"
        "This file is read through @modelcontextprotocol/server-filesystem.\n",
        encoding="utf-8",
    )

    mcp_tool = MCPTool(
        name="filesystem",
        server_command=["npx", "-y", "@modelcontextprotocol/server-filesystem"],
        server_args=[str(demo_root)],
        env={"PYTHONIOENCODING": "utf-8", "NO_COLOR": "1"},
        auto_expand=True,
    )

    expanded_tools = mcp_tool.get_expanded_tools()
    print("[expanded tools]")
    for tool in expanded_tools:
        print(f"- {tool.name}: {tool.description}")

    print("\n[list_tools raw]")
    print(mcp_tool.run({"action": "list_tools"}))

    read_tool = next(tool for tool in expanded_tools if tool.name == "filesystem_read_file")
    print("\n[filesystem_read_file]")
    print(read_tool.run({"path": str(readme)}))


if __name__ == "__main__":
    main()
