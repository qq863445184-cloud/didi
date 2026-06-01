from mcp.server.fastmcp import FastMCP

from app.runner import answer_question

mcp = FastMCP("langgraph-minimal-agent")


@mcp.tool()
def agent_ask(question: str, mode: str = "auto", session_id: str = "default") -> str:
    """Ask the LangGraph agent. Mode can be auto, general, or rag."""
    return answer_question(question, mode, session_id=session_id)


@mcp.tool()
def project_rag(question: str, session_id: str = "default") -> str:
    """Answer a project question using the explicit agentic RAG graph."""
    return answer_question(question, "rag", session_id=session_id)


@mcp.tool()
def coding_plan(question: str, session_id: str = "default") -> str:
    """Create a coding plan without editing files or running commands."""
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


if __name__ == "__main__":
    mcp.run()
