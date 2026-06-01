from pathlib import Path
import json
import re
from typing import Annotated, NotRequired, TypedDict

if __package__ in {None, ""}:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from app.config import configure_stdio, get_settings
from app.context import build_system_context
from app.tools import TOOLS

SYSTEM_PROMPT = (
    "你是一个面向本地 LangGraph 项目的中文 Agent 助手。\n"
    "工作原则：\n"
    "1. 默认使用中文回答，表达清楚、简洁，先给结论再给依据。\n"
    "2. 如果问题涉及当前项目、代码、文件、RAG、工具、配置或实现细节，"
    "优先使用 search_project_docs 或相关文件工具，不要只凭记忆回答。\n"
    "3. 如果使用了 search_project_docs，回答中必须引用工具返回的 Source 标签。\n"
    "4. 如果问题是计算、时间、文件读取等明确工具任务，优先调用对应工具。\n"
    "5. 使用用户画像记忆和项目记忆作为背景，但不要把临时会话内容误当成长期事实。\n"
    "6. 如果证据不足或工具结果不支持结论，不要编造；说明缺少什么，并给出下一步建议。\n"
    "7. 使用 ReAct 循环工作：先理解任务，再在必要时调用工具获取观察结果，"
    "最后基于观察结果回答；不要在最终回答中暴露隐藏推理过程。"
)


class ReActState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    plan: NotRequired[str]
    verifier_note: NotRequired[str]
    verify_attempts: NotRequired[int]
    final_answer: NotRequired[str]


def _last_ai_content(messages: list[AnyMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage) and isinstance(message.content, str):
            return message.content
    return ""


def _format_trace(messages: list[AnyMessage]) -> str:
    lines = []
    for message in messages:
        name = message.__class__.__name__
        content = str(message.content).strip()
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            lines.append(f"{name}: tool_calls={tool_calls}")
        if content:
            lines.append(f"{name}: {content[:1200]}")
    return "\n".join(lines)


def _parse_verifier_note(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return {"sufficient": True, "reason": "Verifier returned non-JSON output."}


def build_graph():
    settings = get_settings()
    base_llm = ChatOpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        model=settings.model_name,
        temperature=0,
    )
    tool_llm = base_llm.bind_tools(TOOLS)

    def planner(state: ReActState, config: RunnableConfig | None = None):
        session_id = (config or {}).get("configurable", {}).get("session_id", "default")
        response = base_llm.invoke(
            [
                SystemMessage(
                    content=build_system_context(
                        (
                            "你是 ReAct planner。请为当前用户任务生成一个极简执行计划。\n"
                            "要求：\n"
                            "1. 最多 3 步。\n"
                            "2. 明确是否需要工具。\n"
                            "3. 只输出计划，不输出答案。"
                        ),
                        session_id=session_id,
                    )
                ),
                *state["messages"],
            ]
        )
        return {"plan": response.content}

    def agent(state: ReActState, config: RunnableConfig | None = None):
        session_id = (config or {}).get("configurable", {}).get("session_id", "default")
        plan = state.get("plan", "")
        verifier_note = state.get("verifier_note", "")
        react_context = "\n".join(
            part
            for part in [
                f"[执行计划]\n{plan}" if plan else "",
                f"[上轮验证反馈]\n{verifier_note}" if verifier_note else "",
            ]
            if part
        )
        messages = [
            SystemMessage(
                content=build_system_context(
                    "\n\n".join(part for part in [SYSTEM_PROMPT, react_context] if part),
                    session_id=session_id,
                )
            ),
            *state["messages"],
        ]
        response = tool_llm.invoke(messages)
        return {"messages": [response]}

    def verifier(state: ReActState, config: RunnableConfig | None = None):
        session_id = (config or {}).get("configurable", {}).get("session_id", "default")
        attempts = state.get("verify_attempts", 0) + 1
        answer = _last_ai_content(state["messages"])
        response = base_llm.invoke(
            [
                SystemMessage(
                    content=build_system_context(
                        (
                            "你是 ReAct verifier。请检查候选答案是否已经完成用户任务。\n"
                            "只返回 JSON："
                            '{"sufficient": true/false, "reason": "简短原因"}\n'
                            "判断重点：是否回答了问题、是否在需要时使用了工具、是否有证据或工具结果支持。"
                        ),
                        session_id=session_id,
                    )
                ),
                HumanMessage(
                    content=(
                        f"用户任务：{state['messages'][0].content}\n\n"
                        f"执行计划：{state.get('plan', '')}\n\n"
                        f"完整执行轨迹：\n{_format_trace(state['messages'])}\n\n"
                        f"候选答案：{answer}"
                    )
                ),
            ]
        )
        return {"verifier_note": response.content, "verify_attempts": attempts}

    def writer(state: ReActState):
        answer = _last_ai_content(state["messages"])
        return {"final_answer": answer}

    def route_after_verifier(state: ReActState) -> str:
        parsed = _parse_verifier_note(state.get("verifier_note", ""))
        if not parsed.get("sufficient", True) and state.get("verify_attempts", 0) < 2:
            return "agent"
        return "writer"

    builder = StateGraph(ReActState)
    builder.add_node("planner", planner)
    builder.add_node("agent", agent)
    builder.add_node("tools", ToolNode(TOOLS))
    builder.add_node("verifier", verifier)
    builder.add_node("writer", writer)
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "agent")
    builder.add_conditional_edges(
        "agent",
        tools_condition,
        {"tools": "tools", "__end__": "verifier"},
    )
    builder.add_edge("tools", "agent")
    builder.add_conditional_edges(
        "verifier",
        route_after_verifier,
        {"agent": "agent", "writer": "writer"},
    )
    builder.add_edge("writer", END)
    return builder.compile()


graph = build_graph()


def run_once(question: str, session_id: str = "default") -> str:
    settings = get_settings()
    result = graph.invoke(
        {
            "messages": [HumanMessage(content=question)],
            "plan": "",
            "verifier_note": "",
            "verify_attempts": 0,
            "final_answer": "",
        },
        config={
            "recursion_limit": settings.recursion_limit,
            "configurable": {"session_id": session_id},
        },
    )
    return result.get("final_answer") or result["messages"][-1].content


if __name__ == "__main__":
    configure_stdio()
    print(run_once("现在北京时间几点？请先用工具获取时间，再用一句话回答。"))
