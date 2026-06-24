from __future__ import annotations

import ast
import operator
import re
from typing import Any

from hello_agents.protocols.a2a.implementation import A2A_AVAILABLE, A2AServer


_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}


def create_simple_a2a_agent() -> A2AServer:
    """Backward-compatible entrypoint for the first simple Chapter 10 demo."""
    return create_calculator_a2a_agent()


def create_calculator_a2a_agent() -> A2AServer:
    """Create a small A2A-style agent that exposes skills to other agents."""
    server = A2AServer(
        name="chapter10-calculator-agent",
        description="A simple A2A agent that can introduce itself and calculate expressions.",
        version="0.1.0",
        capabilities={
            "protocol": "A2A",
            "mode": "simulated" if not A2A_AVAILABLE else "sdk",
            "skills": {
                "introduce": "Describe this agent and its exposed skills.",
                "calculate": "Calculate a safe arithmetic expression.",
            },
        },
    )

    @server.skill("introduce")
    def introduce(_: str) -> str:
        # In A2A, this is similar to the Agent Card: other agents first learn
        # what this agent can do, then decide whether to call a skill.
        return (
            "我是 chapter10-calculator-agent，暴露了 introduce 和 calculate 两个技能。"
            "其它智能体可以通过 A2A 的技能调用方式请求我完成简单计算。"
        )

    @server.skill("calculate")
    def calculate(text: str) -> str:
        expression = _extract_expression(text)
        if not expression:
            return "无法识别可计算表达式，请输入类似 2 + 3 * 4 的算式。"

        value = _safe_eval(expression)
        return f"{expression} = {value:g}"

    return server


def create_writer_a2a_agent() -> A2AServer:
    """Create a second A2A-style agent so the demo shows agent-to-agent routing."""
    server = A2AServer(
        name="chapter10-writer-agent",
        description="A simple A2A agent that can summarize or polish short text.",
        version="0.1.0",
        capabilities={
            "protocol": "A2A",
            "mode": "simulated" if not A2A_AVAILABLE else "sdk",
            "skills": {
                "summarize": "Summarize a short text for another agent.",
                "polish": "Polish a short Chinese sentence.",
            },
        },
    )

    @server.skill("summarize")
    def summarize(text: str) -> str:
        compact = " ".join(text.split())
        if len(compact) <= 48:
            return f"摘要：{compact}"
        return f"摘要：{compact[:48]}..."

    @server.skill("polish")
    def polish(text: str) -> str:
        compact = " ".join(text.split())
        return f"润色稿：{compact}。整体表达更清晰，适合放入第十章示例。"

    return server


def create_a2a_agent_network() -> dict[str, A2AServer]:
    """Create a tiny local A2A network with multiple independent agents."""
    agents = {
        "calculator": create_calculator_a2a_agent(),
        "writer": create_writer_a2a_agent(),
    }
    return agents


def route_a2a_task(agents: dict[str, A2AServer], text: str) -> dict[str, Any]:
    """Route a task to the best A2A agent based on exposed capabilities.

    A production A2A client would discover remote Agent Cards and call skills
    over HTTP. The local demo keeps the same idea but avoids opening ports.
    """
    target_name, skill_name = _select_agent_and_skill(text)
    target_agent = agents[target_name]
    response = call_a2a_skill(target_agent, skill_name, text)
    return {
        "status": response["status"],
        "task": text,
        "target_agent": target_agent.name,
        "skill": skill_name,
        "response": response,
    }


def call_a2a_skill(server: A2AServer, skill_name: str, text: str) -> dict[str, Any]:
    """Call a registered A2A skill directly for tests and local demos.

    The real HTTP path is provided by A2AServer.run(); this helper keeps the
    chapter demo lightweight and deterministic without opening a web port.
    """
    if skill_name not in server.skills:
        return {
            "status": "error",
            "skill": skill_name,
            "error": f"Skill '{skill_name}' not found",
            "available_skills": list(server.skills.keys()),
        }

    try:
        return {
            "status": "success",
            "skill": skill_name,
            "result": server.skills[skill_name](text),
        }
    except Exception as exc:  # pragma: no cover - defensive boundary for demos
        return {"status": "error", "skill": skill_name, "error": str(exc)}


def _select_agent_and_skill(text: str) -> tuple[str, str]:
    lowered = text.lower()
    if "总结" in text or "摘要" in text or "summarize" in lowered:
        return "writer", "summarize"
    if "润色" in text or "polish" in lowered:
        return "writer", "polish"
    if _extract_expression(text) or "计算" in text or "calculate" in lowered:
        return "calculator", "calculate"
    return "writer", "summarize"


def _extract_expression(text: str) -> str:
    """Extract the longest arithmetic-looking segment from natural language."""
    candidates = re.findall(r"[0-9+\-*/().\s^]+", text)
    candidates = [
        item.strip().replace("^", "**")
        for item in candidates
        if re.search(r"\d", item) and re.search(r"[+\-*/^]", item)
    ]
    return max(candidates, key=len, default="")


def _safe_eval(expression: str) -> float:
    """Evaluate arithmetic expressions with an AST allowlist instead of eval."""
    parsed = ast.parse(expression, mode="eval")
    return float(_eval_node(parsed.body))


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _eval_node(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value

    if isinstance(node, ast.BinOp):
        operator_func = _OPERATORS.get(type(node.op))
        if operator_func is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        return float(operator_func(_eval_node(node.left), _eval_node(node.right)))

    raise ValueError(f"Unsupported expression node: {type(node).__name__}")
