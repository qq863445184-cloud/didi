from __future__ import annotations

from typing import Any

from hello_agents import SimpleAgent
from hello_agents.protocols.a2a.implementation import A2A_AVAILABLE, A2AServer

from app.chapter10_a2a_agent import call_a2a_skill


class ReceptionRouterLLM:
    """Deterministic LLM stub for tests; replace with HelloAgentsLLM in real use."""

    def invoke(self, messages: list[dict[str, str]], **_: Any) -> str:
        text = messages[-1]["content"]
        lowered = text.lower()
        if any(word in text for word in ["登录", "报错", "接口", "崩溃", "无法使用"]) or any(
            word in lowered for word in ["login", "error", "api", "crash", "bug"]
        ):
            return "technical"
        if any(word in text for word in ["价格", "套餐", "购买", "试用", "发票"]) or any(
            word in lowered for word in ["price", "pricing", "plan", "trial", "invoice"]
        ):
            return "sales"
        return "human_review"


def create_customer_service_agents(llm: Any | None = None) -> dict[str, A2AServer]:
    """Create a small A2A customer-service network.

    The reception agent owns routing, while domain agents own specialized
    answers. This mirrors the chapter 10.3.4 idea without requiring a running
    HTTP A2A service in tests.
    """
    llm = llm or ReceptionRouterLLM()
    return {
        "reception": _create_reception_agent(llm),
        "technical": _create_technical_agent(),
        "sales": _create_sales_agent(),
        "human_review": _create_human_review_agent(),
    }


def handle_customer_request(
    question: str,
    agents: dict[str, A2AServer] | None = None,
    llm: Any | None = None,
) -> dict[str, Any]:
    """Route one customer request through multiple A2A agents."""
    agents = agents or create_customer_service_agents(llm=llm)
    trace: list[dict[str, Any]] = []

    trace.append({"stage": "customer.request", "question": question})

    classification = call_a2a_skill(agents["reception"], "classify", question)
    route = classification["result"]
    trace.append(
        {
            "stage": "reception.classify",
            "agent": agents["reception"].name,
            "route": route,
        }
    )

    target_key = _route_to_agent_key(route)
    skill_name = "solve" if target_key == "technical" else "consult"
    if target_key == "human_review":
        skill_name = "review"

    trace.append(
        {
            "stage": "a2a.delegate",
            "target_agent": agents[target_key].name,
            "skill": skill_name,
        }
    )

    answer = call_a2a_skill(agents[target_key], skill_name, question)
    trace.append(
        {
            "stage": "a2a.response",
            "target_agent": agents[target_key].name,
            "status": answer["status"],
        }
    )

    return {
        "status": answer["status"],
        "question": question,
        "route": route,
        "target_agent": agents[target_key].name,
        "skill": skill_name,
        "answer": answer["result"],
        "trace": trace,
    }


def list_customer_service_agent_cards(agents: dict[str, A2AServer]) -> dict[str, dict[str, Any]]:
    """Expose all Agent Cards so another agent can discover capabilities."""
    return {name: agent.get_info() for name, agent in agents.items()}


def _create_reception_agent(llm: Any) -> A2AServer:
    server = A2AServer(
        name="customer-reception-agent",
        description="Uses a SimpleAgent receptionist to route customer requests.",
        version="0.1.0",
        capabilities={
            "protocol": "A2A",
            "mode": "simulated" if not A2A_AVAILABLE else "sdk",
            "skills": {"classify": "Use a receptionist SimpleAgent to classify a request."},
        },
    )
    receptionist = SimpleAgent(
        name="接待员",
        llm=llm,
        system_prompt=(
            "你是客服接待员，负责：\n"
            "1. 分析客户问题类型，并且必须只返回一个英文标签。\n"
            "2. 如果是登录失败、报错、接口、崩溃、无法使用、token、API、错误码，返回 technical。\n"
            "3. 如果是价格、套餐、购买、试用、发票、合同、商务合作，返回 sales。\n"
            "4. 如果问题模糊、涉及投诉、退款争议、隐私、安全合规或明确要求人工，返回 human_review。\n"
            "5. 输出只能是 technical、sales、human_review 三者之一，不要解释，不要加标点。\n"
            "示例：\n"
            "用户：登录一直报错 token 无效\n"
            "你：technical\n"
            "用户：有没有试用套餐\n"
            "你：sales\n"
            "用户：我要找人工投诉\n"
            "你：human_review"
        ),
    )

    @server.skill("classify")
    def classify(text: str) -> str:
        raw_route = receptionist.run(text)
        server.last_raw_route = raw_route
        return _normalize_route(raw_route)

    return server


def _create_technical_agent() -> A2AServer:
    server = A2AServer(
        name="technical-support-agent",
        description="Handles technical support requests.",
        version="0.1.0",
        capabilities={
            "protocol": "A2A",
            "mode": "simulated" if not A2A_AVAILABLE else "sdk",
            "skills": {"solve": "Provide troubleshooting steps for technical issues."},
        },
    )

    @server.skill("solve")
    def solve(text: str) -> str:
        if "登录" in text or "login" in text.lower():
            return "技术支持：请先确认账号状态、重置密码，并检查是否开启了多因素认证。"
        if "接口" in text or "api" in text.lower():
            return "技术支持：请检查 API Key、Base URL、模型名和请求体字段，并保留错误码用于排查。"
        return "技术支持：请提供复现步骤、错误日志、发生时间和影响范围，我们会继续定位。"

    return server


def _create_sales_agent() -> A2AServer:
    server = A2AServer(
        name="sales-advisor-agent",
        description="Handles pricing, plan, trial, and invoice questions.",
        version="0.1.0",
        capabilities={
            "protocol": "A2A",
            "mode": "simulated" if not A2A_AVAILABLE else "sdk",
            "skills": {"consult": "Answer pricing and purchase questions."},
        },
    )

    @server.skill("consult")
    def consult(text: str) -> str:
        if "试用" in text or "trial" in text.lower():
            return "销售顾问：可以先开通试用版，确认并发量、调用额度和私有化需求后再选择套餐。"
        if "发票" in text or "invoice" in text.lower():
            return "销售顾问：支持企业发票，请提供抬头、税号、订单号和开票邮箱。"
        return "销售顾问：建议按调用量、并发要求、SLA 和是否需要私有化部署选择套餐。"

    return server


def _create_human_review_agent() -> A2AServer:
    server = A2AServer(
        name="human-review-agent",
        description="Escalates ambiguous or sensitive customer requests to a human operator.",
        version="0.1.0",
        capabilities={
            "protocol": "A2A",
            "mode": "simulated" if not A2A_AVAILABLE else "sdk",
            "skills": {"review": "Prepare a human handoff summary."},
        },
    )

    @server.skill("review")
    def review(text: str) -> str:
        return f"人工复核：该问题需要人工客服介入。已记录用户诉求：{text}"

    return server


def _route_to_agent_key(route: str) -> str:
    if route in {"technical", "sales", "human_review"}:
        return route
    return "human_review"


def _normalize_route(raw_route: str) -> str:
    route = raw_route.strip().lower()
    if "technical" in route or "技术" in route:
        return "technical"
    if "sales" in route or "销售" in route:
        return "sales"
    return "human_review"
