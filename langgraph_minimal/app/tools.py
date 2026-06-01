from ast import Add, BinOp, Constant, Div, Expression, Mult, Sub, UAdd, USub, UnaryOp, parse
from datetime import datetime, timedelta, timezone
from pathlib import Path

from langchain_core.tools import tool

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HIDDEN_ENTRIES = {".env", ".venv", "__pycache__"}


@tool
def get_current_time() -> str:
    """Get the current date and time in UTC+8."""
    now = datetime.now(timezone(timedelta(hours=8)))
    return now.strftime("%Y-%m-%d %H:%M:%S UTC+08:00")


def _eval_expr(node) -> float:
    if isinstance(node, Expression):
        return _eval_expr(node.body)
    if isinstance(node, BinOp):
        left = _eval_expr(node.left)
        right = _eval_expr(node.right)
        if isinstance(node.op, Add):
            return left + right
        if isinstance(node.op, Sub):
            return left - right
        if isinstance(node.op, Mult):
            return left * right
        if isinstance(node.op, Div):
            return left / right
        raise ValueError("Unsupported operator")
    if isinstance(node, UnaryOp):
        value = _eval_expr(node.operand)
        if isinstance(node.op, UAdd):
            return +value
        if isinstance(node.op, USub):
            return -value
        raise ValueError("Unsupported unary operator")
    if isinstance(node, Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("Only numbers are supported")
        return float(node.value)
    raise ValueError("Unsupported expression")


@tool
def calculate(expression: str) -> str:
    """Safely evaluate a basic arithmetic expression with +, -, *, /, and parentheses."""
    try:
        tree = parse(expression, mode="eval")
        result = _eval_expr(tree)
    except Exception as exc:
        return f"Calculation error: {exc}"

    if result.is_integer():
        return str(int(result))
    return str(result)


def _resolve_project_path(path: str) -> Path:
    target = (PROJECT_ROOT / path).resolve()
    if target != PROJECT_ROOT and PROJECT_ROOT not in target.parents:
        raise ValueError("Path must stay inside the langgraph_minimal project.")
    return target


def _is_hidden(path: Path) -> bool:
    return any(part in HIDDEN_ENTRIES for part in path.relative_to(PROJECT_ROOT).parts)


@tool
def list_files(path: str = ".") -> str:
    """List files and directories inside the project."""
    try:
        target = _resolve_project_path(path)
    except ValueError as exc:
        return f"Path error: {exc}"

    if not target.exists():
        return f"Path does not exist: {path}"
    if target.is_file():
        return str(target.relative_to(PROJECT_ROOT))

    entries = []
    for child in sorted(target.iterdir()):
        if _is_hidden(child):
            continue
        marker = "/" if child.is_dir() else ""
        entries.append(f"{child.relative_to(PROJECT_ROOT)}{marker}")
    return "\n".join(entries) if entries else "(empty directory)"


@tool
def read_text_file(path: str, max_chars: int = 4000) -> str:
    """Read a UTF-8 text file inside the project."""
    try:
        target = _resolve_project_path(path)
    except ValueError as exc:
        return f"Path error: {exc}"

    if _is_hidden(target):
        return "Path error: this file is hidden from the agent."
    if not target.exists():
        return f"File does not exist: {path}"
    if not target.is_file():
        return f"Path is not a file: {path}"

    limit = max(1, min(max_chars, 20000))
    text = target.read_text(encoding="utf-8", errors="replace")
    if len(text) > limit:
        return text[:limit] + "\n...[truncated]"
    return text


@tool
def search_project_docs(query: str, top_k: int = 5) -> str:
    """Search project documents and code snippets, returning cited chunks."""
    from app.rag import format_search_results

    return format_search_results(query=query, top_k=top_k)


TOOLS = [get_current_time, calculate, list_files, read_text_file, search_project_docs]
