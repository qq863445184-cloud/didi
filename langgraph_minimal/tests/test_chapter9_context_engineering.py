from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.context_engineering import (
    ContextBuilder,
    ContextConfig,
    ContextPacket,
    ContextQualityReport,
    estimate_tokens,
    lexical_relevance,
)


def test_context_builder_keeps_system_and_selects_relevant_packets():
    now = datetime(2026, 6, 17, 12, 0, tzinfo=timezone.utc)
    builder = ContextBuilder(
        config=ContextConfig(max_tokens=240, reserve_ratio=0.1, min_relevance=0.05),
        now=now,
    )

    result = builder.build(
        user_query="RAG 上下文工程怎么做？",
        system_instructions="必须基于上下文回答。",
        custom_packets=[
            ContextPacket(
                content="RAG 上下文工程需要先检索证据，再按 token 预算选择。",
                source="rag",
                timestamp=now - timedelta(minutes=5),
                relevance_score=0.9,
            ),
            ContextPacket(
                content="天气很好，适合散步。",
                source="noise",
                timestamp=now,
                relevance_score=0.01,
            ),
        ],
    )

    assert "必须基于上下文回答" in result.context
    assert "RAG 上下文工程" in result.context
    assert "天气很好" not in result.context
    assert any(event["stage"] == "context.gather" for event in result.trace)
    assert any(event["stage"] == "context.select" for event in result.trace)
    assert any(event["stage"] == "context.structure" for event in result.trace)


def test_context_builder_compresses_to_budget():
    now = datetime(2026, 6, 17, 12, 0, tzinfo=timezone.utc)
    builder = ContextBuilder(
        config=ContextConfig(max_tokens=90, reserve_ratio=0.0, min_relevance=0.0),
        now=now,
    )

    result = builder.build(
        user_query="解释上下文压缩",
        system_instructions="保留核心规则。",
        custom_packets=[
            ContextPacket(
                content="上下文压缩要在有限 token 预算内保留最相关信息。" * 30,
                source="chapter9",
                relevance_score=0.95,
            )
        ],
    )

    assert result.total_tokens <= 90
    assert "保留核心规则" in result.context
    assert "[compressed]" in result.context
    assert any(event["stage"] == "context.compress" for event in result.trace)


def test_context_builder_structures_explicit_sections():
    now = datetime(2026, 6, 17, 12, 0, tzinfo=timezone.utc)
    builder = ContextBuilder(
        config=ContextConfig(max_tokens=400, reserve_ratio=0.0, min_relevance=0.0),
        now=now,
    )

    result = builder.build(
        user_query="如何组织上下文？",
        system_instructions="你是上下文工程助手。",
        conversation_history=[
            {
                "role": "user",
                "content": "上一轮讨论了第八章记忆系统。",
                "timestamp": now,
            }
        ],
        custom_packets=[
            ContextPacket(
                content="RAG 检索到了上下文工程的 GSSC 流程。",
                source="chapter9-note",
                relevance_score=0.9,
                metadata={"section": "evidence"},
            )
        ],
        output_instructions="用分点说明。",
    )

    assert "[Role & Policies]" in result.context
    assert "[Task]" in result.context
    assert "[Evidence]" in result.context
    assert "[Context]" in result.context
    assert "[Output]" in result.context
    assert "[State]" not in result.context
    assert "RAG 检索到了上下文工程的 GSSC 流程" in result.context
    assert "上一轮讨论了第八章记忆系统" in result.context


def test_lexical_relevance_and_token_estimator_are_deterministic():
    assert lexical_relevance("RAG 上下文工程需要证据", "RAG 证据") > 0
    assert lexical_relevance("天气很好", "RAG 证据") == 0
    assert estimate_tokens("RAG 上下文工程") == estimate_tokens("RAG 上下文工程")


def test_context_builder_reports_quality_metrics_and_suggestions():
    builder = ContextBuilder(
        config=ContextConfig(max_tokens=500, reserve_ratio=0.0, min_relevance=0.0)
    )

    result = builder.build(
        user_query="如何重构 load_user？",
        system_instructions="基于证据回答。",
        custom_packets=[
            ContextPacket(
                content="app/service.py 的 load_user 需要拆出 validate_user_id 和 fetch_user。",
                source="terminal",
                relevance_score=0.9,
                metadata={"section": "evidence"},
            ),
            ContextPacket(
                content="天气很好，午饭可以吃面。",
                source="noise",
                relevance_score=0.0,
                metadata={"section": "context"},
            ),
        ],
        output_instructions="给出下一步计划。",
    )

    assert isinstance(result.quality, ContextQualityReport)
    assert 0.0 <= result.quality.information_density <= 1.0
    assert 0.0 <= result.quality.relevance <= 1.0
    assert 0.0 <= result.quality.completeness <= 1.0
    assert result.quality.overall_score <= 1.0
    assert any("移除或降权低相关" in suggestion for suggestion in result.quality.suggestions)
    assert any(event["stage"] == "context.quality" for event in result.trace)


def test_context_builder_quality_warns_when_context_is_incomplete():
    builder = ContextBuilder(config=ContextConfig(max_tokens=220, reserve_ratio=0.0))

    result = builder.build(
        user_query="如何重构 load_user？",
        system_instructions="你是维护助手。",
        custom_packets=[],
    )

    assert result.quality.completeness < 0.7
    assert any("补充 Evidence" in suggestion for suggestion in result.quality.suggestions)
