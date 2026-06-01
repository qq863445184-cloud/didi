from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

class RouterState(TypedDict):
    question: str
    session_id: str
    route: str
    answer: str


RAG_KEYWORDS = {
    "rag",
    "source",
    "引用",
    "来源",
    "项目",
    "文档",
    "代码",
    "文件",
    "readme",
    "graph.py",
    "rag.py",
    "tools.py",
    "config.py",
    "cli.py",
    "当前 agent",
    "这个 agent",
}


def router(state: RouterState) -> dict:
    question = state["question"].lower()
    route = "rag" if any(keyword in question for keyword in RAG_KEYWORDS) else "general"
    return {"route": route}


def general_agent(state: RouterState) -> dict:
    from app.graph import run_once

    return {"answer": run_once(state["question"], session_id=state.get("session_id", "default"))}


def rag_agent(state: RouterState) -> dict:
    from app.rag_graph import answer_with_rag

    return {
        "answer": answer_with_rag(
            state["question"],
            session_id=state.get("session_id", "default"),
        )
    }


def route_after_router(state: RouterState) -> Literal["general", "rag"]:
    return "rag" if state["route"] == "rag" else "general"


builder = StateGraph(RouterState)
builder.add_node("router", router)
builder.add_node("general", general_agent)
builder.add_node("rag", rag_agent)
builder.add_edge(START, "router")
builder.add_conditional_edges(
    "router",
    route_after_router,
    {"general": "general", "rag": "rag"},
)
builder.add_edge("general", END)
builder.add_edge("rag", END)

router_graph = builder.compile()


def answer(question: str, session_id: str = "default") -> str:
    result = router_graph.invoke(
        {"question": question, "session_id": session_id, "route": "", "answer": ""}
    )
    return result["answer"]
