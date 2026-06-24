from __future__ import annotations

from app.chapter10_a2a_agent import call_a2a_skill, create_simple_a2a_agent


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
