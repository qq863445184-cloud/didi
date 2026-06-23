from __future__ import annotations

from hello_agents.tools import MCPTool


def test_external_filesystem_mcp_tool_lists_and_reads_files(tmp_path):
    (tmp_path / "README.md").write_text(
        "External filesystem MCP server is connected.\n",
        encoding="utf-8",
    )

    mcp_tool = MCPTool(
        name="filesystem",
        server_command=["npx", "-y", "@modelcontextprotocol/server-filesystem"],
        server_args=[str(tmp_path)],
        auto_expand=True,
    )
    expanded = mcp_tool.get_expanded_tools()
    names = {tool.name for tool in expanded}

    assert "filesystem_read_file" in names
    assert "filesystem_list_directory" in names

    read_tool = next(tool for tool in expanded if tool.name == "filesystem_read_file")
    result = read_tool.run({"path": str(tmp_path / "README.md")})

    assert "External filesystem MCP server is connected" in result
