from __future__ import annotations

from app.chapter10_customer_service_a2a import (
    ReceptionRouterLLM,
    create_customer_service_agents,
    handle_customer_request,
    list_customer_service_agent_cards,
)


class RecordingRouterLLM(ReceptionRouterLLM):
    def __init__(self):
        self.calls = []

    def invoke(self, messages, **kwargs):
        self.calls.append(messages)
        return super().invoke(messages, **kwargs)


def test_customer_service_exposes_multiple_agent_cards():
    agents = create_customer_service_agents()
    cards = list_customer_service_agent_cards(agents)

    assert set(cards) == {"reception", "technical", "sales", "human_review"}
    assert cards["reception"]["protocol"] == "A2A"
    assert "classify" in cards["reception"]["skills"]


def test_customer_service_routes_technical_issue():
    result = handle_customer_request("我登录一直报错，提示 token 无效。")

    assert result["route"] == "technical"
    assert result["target_agent"] == "technical-support-agent"
    assert result["skill"] == "solve"
    assert "技术支持" in result["answer"]


def test_customer_service_routes_sales_question():
    result = handle_customer_request("你们有没有试用套餐，后续怎么购买？")

    assert result["route"] == "sales"
    assert result["target_agent"] == "sales-advisor-agent"
    assert result["skill"] == "consult"
    assert "销售顾问" in result["answer"]


def test_customer_service_escalates_ambiguous_request():
    result = handle_customer_request("我这个问题比较复杂，想找人工客服处理。")

    assert result["route"] == "human_review"
    assert result["target_agent"] == "human-review-agent"
    assert result["skill"] == "review"
    assert "人工复核" in result["answer"]


def test_customer_service_trace_records_a2a_handoff():
    result = handle_customer_request("API 接口返回 401 error")
    stages = [item["stage"] for item in result["trace"]]

    assert stages == [
        "customer.request",
        "reception.classify",
        "a2a.delegate",
        "a2a.response",
    ]


def test_receptionist_routing_is_done_by_simple_agent_llm():
    llm = RecordingRouterLLM()

    result = handle_customer_request("你们有没有试用套餐？", llm=llm)

    assert result["route"] == "sales"
    assert len(llm.calls) == 1
    messages = llm.calls[0]
    assert messages[0]["role"] == "system"
    assert "你是客服接待员" in messages[0]["content"]
    assert messages[-1]["role"] == "user"
    assert "试用套餐" in messages[-1]["content"]
