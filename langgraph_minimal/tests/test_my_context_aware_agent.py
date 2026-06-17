from __future__ import annotations

from app.context_engineering import ContextConfig, ContextPacket
from app.my_context_aware_agent import MyContextAwareAgent


class FakeContextLLM:
    provider = "fake"

    def __init__(self) -> None:
        self.last_messages = []

    def invoke(self, messages, **kwargs):
        self.last_messages = messages
        content = messages[-1]["content"]
        if "GSSC" in content:
            return "上下文工程流程是 Gather、Select、Structure、Compress。"
        return "已基于上下文回答。"

    def stream_invoke(self, messages, **kwargs):
        self.last_messages = messages
        yield "流式"
        yield "上下文回答"


def test_context_aware_agent_builds_context_before_llm():
    llm = FakeContextLLM()
    agent = MyContextAwareAgent(
        name="上下文助手",
        llm=llm,
        system_prompt="你是第九章上下文工程助手。",
        context_config=ContextConfig(max_tokens=240),
    )

    response = agent.run(
        "什么是 GSSC？",
        custom_packets=[
            ContextPacket(
                source="chapter9",
                content="GSSC 表示 Gather、Select、Structure、Compress。",
                relevance_score=0.95,
            )
        ],
    )

    assert "Gather、Select、Structure、Compress" in response
    assert agent.last_context_result is not None
    assert "[Role & Policies]" in llm.last_messages[-1]["content"]
    assert "[Evidence]" in llm.last_messages[-1]["content"]
    assert "[Context]" not in llm.last_messages[-1]["content"]
    assert any(event["stage"] == "context.build" for event in agent.trace_events)
    assert agent.trace_events[-1]["stage"] == "agent.answer"


def test_context_aware_agent_uses_conversation_history_on_next_turn():
    llm = FakeContextLLM()
    agent = MyContextAwareAgent(
        name="上下文助手",
        llm=llm,
        context_config=ContextConfig(max_tokens=320),
    )

    agent.run("第一轮问题", custom_packets=[ContextPacket(content="第一轮资料", relevance_score=0.9)])
    agent.run("第二轮问题", custom_packets=[ContextPacket(content="第二轮资料", relevance_score=0.9)])

    prompt = llm.last_messages[-1]["content"]
    assert "user: 第一轮问题" in prompt
    assert "assistant:" in prompt
    assert len(agent.get_history()) == 4


def test_context_aware_agent_stream_run_keeps_history():
    llm = FakeContextLLM()
    agent = MyContextAwareAgent(name="上下文助手", llm=llm)

    response = "".join(
        agent.stream_run(
            "流式问题",
            custom_packets=[ContextPacket(content="流式上下文", relevance_score=0.9)],
        )
    )

    assert response == "流式上下文回答"
    assert len(agent.get_history()) == 2
    assert agent.trace_events[-1]["stage"] == "agent.answer_stream"
