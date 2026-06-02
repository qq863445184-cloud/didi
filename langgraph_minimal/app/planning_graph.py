import os
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from app.coding_prompts import (
    CODING_PLANNER_PROMPT,
    CODING_SYSTEM_PROMPT,
    INTAKE_PROMPT,
    LOCATOR_PROMPT,
    REPO_INSPECTOR_PROMPT,
)
from app.config import get_settings
from app.context import build_system_context
from app.rag import format_search_results


class PlanningState(TypedDict):
    question: str
    session_id: str
    intake: str
    repo_context: str
    locations: str
    plan: str


def _llm() -> ChatOpenAI:
    settings = get_settings()
    timeout = int(os.getenv("PLAN_LLM_TIMEOUT", "60"))
    return ChatOpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        model=settings.model_name,
        temperature=0,
        timeout=timeout,
        max_retries=0,
    )


def collect_context(state: PlanningState) -> dict:
    evidence = format_search_results(state["question"], top_k=5)
    return {
        "intake": "由 planner 根据用户任务整理。",
        "repo_context": evidence,
        "locations": evidence,
    }


def planner(state: PlanningState) -> dict:
    if os.getenv("PLAN_LLM_ENABLED", "false").strip().lower() != "true":
        return {"plan": _fallback_plan(state, RuntimeError("PLAN_LLM_ENABLED=false"))}

    try:
        response = _llm().invoke(
            [
                SystemMessage(
                    content=build_system_context(
                        "\n\n".join(
                            [
                                CODING_SYSTEM_PROMPT,
                                INTAKE_PROMPT,
                                REPO_INSPECTOR_PROMPT,
                                LOCATOR_PROMPT,
                                CODING_PLANNER_PROMPT,
                                (
                                    "这是计划模式。只输出计划，不修改文件，不运行命令。"
                                    "计划必须可供用户审核后再执行。"
                                ),
                            ]
                        ),
                        session_id=state.get("session_id", "default"),
                    )
                ),
                HumanMessage(
                    content=(
                        f"用户任务：{state['question']}\n\n"
                        f"项目检索证据：\n{state['repo_context']}\n\n"
                        "请输出：任务理解、相关文件、修改计划、验证计划、风险。"
                    )
                ),
            ]
        )
        return {"plan": response.content}
    except Exception as exc:
        return {"plan": _fallback_plan(state, exc)}


def _fallback_plan(state: PlanningState, exc: Exception) -> str:
    evidence = state.get("repo_context", "")
    sources = [
        line.replace("Source: ", "").strip()
        for line in evidence.splitlines()
        if line.startswith("Source: ")
    ][:8]
    source_block = "\n".join(f"- {source}" for source in sources) or "- 暂无可靠来源"
    return (
        "## 任务理解\n"
        f"- 目标：{state['question']}\n"
        "- 模式：计划模式，只生成计划，不修改文件，不运行命令。\n"
        f"- 备注：LLM 计划未启用或生成失败，已使用本地 fallback。原因：{exc}\n\n"
        "## 相关文件/证据\n"
        f"{source_block}\n\n"
        "## 修改计划\n"
        "1. 先阅读上述相关文件，确认当前实现和调用关系。\n"
        "2. 定位与任务目标直接相关的函数、类、配置或测试。\n"
        "3. 制定最小 diff，只修改完成目标所需的位置。\n"
        "4. 如涉及行为变化，补充或更新对应测试/文档。\n\n"
        "## 验证计划\n"
        "1. 运行 `python -m compileall app`。\n"
        "2. 根据实际修改运行相关 CLI smoke test。\n"
        "3. 如果失败，保留错误日志并进入 debug -> fix 循环。\n\n"
        "## 风险\n"
        "- fallback 计划没有经过模型深度推理，执行前需要人工或 agent 再确认具体文件内容。\n"
    )


builder = StateGraph(PlanningState)
builder.add_node("collect_context", collect_context)
builder.add_node("planner", planner)
builder.add_edge(START, "collect_context")
builder.add_edge("collect_context", "planner")
builder.add_edge("planner", END)

planning_graph = builder.compile()


def make_plan(question: str, session_id: str = "default") -> str:
    result = planning_graph.invoke(
        {
            "question": question,
            "session_id": session_id,
            "intake": "",
            "repo_context": "",
            "locations": "",
            "plan": "",
        }
    )
    return result["plan"]
