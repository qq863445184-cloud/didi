from __future__ import annotations

from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from app.note_tool import NoteTool
from app.terminal_tool import TerminalTool


def create_mcp_server(
    *,
    root: str | Path | None = None,
    note_path: str | Path | None = None,
    user_id: str = "default",
    include_agent_tools: bool = True,
) -> FastMCP:
    """Create the Chapter 10 MCP server.

    MCP Server 的职责是把本地 Tool 以协议化方式暴露出去，让外部
    Agent 可以先 list_tools 发现能力，再 call_tool 调用能力。
    """

    workspace_root = Path(root or Path.cwd()).expanduser().resolve()
    notes_file = Path(note_path or workspace_root / "memory_data" / "mcp_notes.jsonl")
    terminal_tool = TerminalTool(root=workspace_root)
    note_tool = NoteTool(path=notes_file, user_id=user_id)
    mcp = FastMCP("langgraph-minimal-agent")

    register_local_tools(mcp, terminal_tool=terminal_tool, note_tool=note_tool)
    if include_agent_tools:
        register_agent_tools(mcp)
    return mcp


def register_local_tools(
    mcp: FastMCP,
    *,
    terminal_tool: TerminalTool,
    note_tool: NoteTool,
) -> None:
    """Register our custom Chapter 9 tools as MCP tools."""

    @mcp.tool(
        name="terminal_run",
        description=(
            "Run a read-only filesystem action. Supported action values: "
            "pwd, list, read, search, stat."
        ),
    )
    def terminal_run(
        action: str,
        path: str = ".",
        query: str = "",
        pattern: str = "*",
        limit: int = 20,
        max_bytes: int | None = None,
    ) -> str:
        """Use TerminalTool through MCP for safe codebase inspection."""

        parameters: dict[str, Any] = {
            "action": action,
            "path": path,
            "query": query,
            "pattern": pattern,
            "limit": limit,
        }
        if max_bytes is not None:
            parameters["max_bytes"] = max_bytes
        return terminal_tool.run(parameters)

    @mcp.tool(
        name="note_run",
        description=(
            "Run a persistent note action. Supported action values: "
            "add, search, list, update, remove, stats."
        ),
    )
    def note_run(
        action: str,
        content: str = "",
        title: str = "",
        query: str = "",
        note_id: str = "",
        tags: list[str] | str | None = None,
        importance: float = 0.5,
        limit: int = 5,
    ) -> str:
        """Use NoteTool through MCP for task notes and long-running records."""

        return note_tool.run(
            {
                "action": action,
                "content": content,
                "title": title,
                "query": query,
                "note_id": note_id,
                "tags": tags,
                "importance": importance,
                "limit": limit,
            }
        )


def register_agent_tools(mcp: FastMCP) -> None:
    """Register the older LangGraph demo tools."""

    @mcp.tool()
    def agent_ask(question: str, mode: str = "auto", session_id: str = "default") -> str:
        """Ask the LangGraph agent. Mode can be auto, general, or rag."""

        from app.runner import answer_question

        return answer_question(question, mode, session_id=session_id)

    @mcp.tool()
    def project_rag(question: str, session_id: str = "default") -> str:
        """Answer a project question using the explicit agentic RAG graph."""

        from app.runner import answer_question

        return answer_question(question, "rag", session_id=session_id)

    @mcp.tool()
    def coding_plan(question: str, session_id: str = "default") -> str:
        """Create a coding plan without editing files or running commands."""

        from app.runner import answer_question

        return answer_question(question, "plan", session_id=session_id)

    @mcp.tool()
    def search_project_docs(query: str, top_k: int = 5) -> str:
        """Search project documents and return cited chunks."""

        from app.rag import format_search_results

        return format_search_results(query, top_k=top_k)

    @mcp.tool()
    def calculate(expression: str) -> str:
        """Evaluate a basic arithmetic expression."""

        from app.tools import calculate as calculate_tool

        return calculate_tool.invoke({"expression": expression})

    @mcp.tool()
    def get_current_time() -> str:
        """Get the current time in UTC+8."""

        from app.tools import get_current_time as time_tool

        return time_tool.invoke({})

    @mcp.tool()
    def get_weather(city: str = "北京") -> str:
        """Get today's current weather for a city."""

        from app.tools import get_weather as weather_tool

        return weather_tool.invoke({"city": city})


mcp = create_mcp_server(root=Path(__file__).resolve().parents[1])


if __name__ == "__main__":
    mcp.run()
