from __future__ import annotations

from app.chapter10_a2a_agent import (
    call_a2a_skill,
    create_a2a_agent_network,
    create_simple_a2a_agent,
    route_a2a_task,
)


def test_simple_a2a_agent_exposes_agent_card_and_skills():
    agent = create_simple_a2a_agent()
    info = agent.get_info()

    assert info["name"] == "chapter10-calculator-agent"
    assert info["protocol"] == "A2A"
    assert {"introduce", "calculate"}.issubset(set(info["skills"]))


def test_simple_a2a_agent_calls_calculate_skill():
    agent = create_simple_a2a_agent()

    result = call_a2a_skill(agent, "calculate", "请计算 2 + 3 * 4")

    assert result["status"] == "success"
    assert result["skill"] == "calculate"
    assert result["result"] == "2 + 3 * 4 = 14"


def test_simple_a2a_agent_reports_unknown_skill():
    agent = create_simple_a2a_agent()

    result = call_a2a_skill(agent, "unknown", "hello")

    assert result["status"] == "error"
    assert "available_skills" in result
    assert "calculate" in result["available_skills"]


def test_a2a_network_exposes_multiple_agent_cards():
    agents = create_a2a_agent_network()

    assert set(agents) == {"calculator", "writer"}
    assert agents["calculator"].get_info()["name"] == "chapter10-calculator-agent"
    assert agents["writer"].get_info()["name"] == "chapter10-writer-agent"


def test_a2a_router_delegates_calculation_to_calculator_agent():
    agents = create_a2a_agent_network()

    result = route_a2a_task(agents, "请计算 (8 + 4) / 3")

    assert result["status"] == "success"
    assert result["target_agent"] == "chapter10-calculator-agent"
    assert result["skill"] == "calculate"
    assert result["response"]["result"] == "(8 + 4) / 3 = 4"


def test_a2a_router_delegates_summary_to_writer_agent():
    agents = create_a2a_agent_network()

    result = route_a2a_task(agents, "总结：A2A 通过 Agent Card 发现能力并进行技能调用。")

    assert result["status"] == "success"
    assert result["target_agent"] == "chapter10-writer-agent"
    assert result["skill"] == "summarize"
    assert result["response"]["result"].startswith("摘要：")
