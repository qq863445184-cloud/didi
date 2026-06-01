import json
import re
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from app.config import get_settings
from app.context import build_rag_writer_context
from app.rag import format_search_results


class RagState(TypedDict):
    question: str
    query: str
    session_id: str
    attempts: int
    evidence: str
    sufficient: bool
    verification: str
    final_answer: str


def _llm() -> ChatOpenAI:
    settings = get_settings()
    return ChatOpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        model=settings.model_name,
        temperature=0,
    )


def retrieve(state: RagState) -> dict:
    query = state.get("query") or state["question"]
    evidence = format_search_results(query=query, top_k=get_settings().rag_final_top_k)
    return {
        "query": query,
        "attempts": state.get("attempts", 0) + 1,
        "evidence": evidence,
    }


def verify(state: RagState) -> dict:
    settings = get_settings()
    response = _llm().invoke(
        [
            SystemMessage(
                content=(
                    "你是 RAG 证据验证器。\n"
                    "任务：判断 [检索证据] 是否足以回答 [当前问题]。\n"
                    "只返回 JSON，不要输出解释性文本。JSON 字段必须包含：\n"
                    "{\n"
                    '  "sufficient": true/false,\n'
                    '  "reason": "简要说明判断原因",\n'
                    '  "next_query": "如果证据不足，给出更适合继续检索的新查询；否则为空字符串"\n'
                    "}\n"
                    "判断标准：\n"
                    "1. 证据是否直接相关。\n"
                    "2. 证据是否包含足够细节。\n"
                    "3. 证据是否包含可引用的 Source。\n"
                    "4. 证据之间是否存在冲突。\n"
                    "5. 如果证据不足且仍可重试，next_query 必须更具体。"
                )
            ),
            HumanMessage(
                content=(
                    f"Question: {state['question']}\n"
                    f"Current query: {state['query']}\n"
                    f"Attempt: {state['attempts']} / {settings.rag_max_attempts}\n\n"
                    f"Evidence:\n{state['evidence'][:12000]}"
                )
            ),
        ]
    )
    parsed = _parse_json(response.content)
    sufficient = bool(parsed.get("sufficient", True))
    reason = str(parsed.get("reason", ""))
    next_query = str(parsed.get("next_query", "")).strip()

    if state["attempts"] >= settings.rag_max_attempts:
        sufficient = True
    if not next_query:
        next_query = state["query"]

    return {
        "sufficient": sufficient,
        "verification": reason,
        "query": next_query,
    }


def writer(state: RagState) -> dict:
    system_context, human_context = build_rag_writer_context(
        base_instruction=(
            "你是一个基于证据回答问题的中文 RAG writer。\n"
            "回答规则：\n"
            "1. 只能基于 [检索证据] 回答，不要使用没有证据支持的项目事实。\n"
            "2. 必须使用证据中的原始 Source 标签引用来源。\n"
            "3. 如果证据互相冲突，指出冲突并分别引用。\n"
            "4. 如果证据不足，明确说明“不足以确认”，并说明缺少什么。\n"
            "5. 回答优先结构化：先给结论，再给依据，最后给必要的下一步建议。\n"
            "6. 不要泄露隐藏配置、密钥或未被证据支持的敏感信息。"
        ),
        question=state["question"],
        verification=state.get("verification", ""),
        evidence=state["evidence"],
        session_id=state.get("session_id", "default"),
    )
    response = _llm().invoke(
        [
            SystemMessage(content=system_context),
            HumanMessage(content=human_context),
        ]
    )
    return {"final_answer": response.content}


def route_after_verify(state: RagState) -> str:
    if state.get("sufficient", True):
        return "writer"
    return "retrieve"


def route_after_retrieve(state: RagState) -> str:
    return "verify" if get_settings().rag_verify_enabled else "writer"


def _parse_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return {"sufficient": True, "reason": "Verifier returned non-JSON output.", "next_query": ""}


builder = StateGraph(RagState)
builder.add_node("retrieve", retrieve)
builder.add_node("verify", verify)
builder.add_node("writer", writer)
builder.add_edge(START, "retrieve")
builder.add_conditional_edges(
    "retrieve",
    route_after_retrieve,
    {"verify": "verify", "writer": "writer"},
)
builder.add_conditional_edges(
    "verify",
    route_after_verify,
    {"retrieve": "retrieve", "writer": "writer"},
)
builder.add_edge("writer", END)

rag_graph = builder.compile()


def answer_with_rag(question: str, session_id: str = "default") -> str:
    result = rag_graph.invoke(
        {
            "question": question,
            "query": question,
            "session_id": session_id,
            "attempts": 0,
            "evidence": "",
            "sufficient": False,
            "verification": "",
            "final_answer": "",
        },
        config={"recursion_limit": max(6, get_settings().rag_max_attempts * 4)},
    )
    return result["final_answer"]
