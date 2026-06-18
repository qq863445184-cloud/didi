from __future__ import annotations

import json
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.context_engineering import ContextBuilder, ContextConfig, ContextPacket


def build_demo() -> ContextBuilder:
    return ContextBuilder(
        config=ContextConfig(
            max_tokens=360,
            reserve_ratio=0.15,
            min_relevance=0.05,
        ),
        now=datetime(2026, 6, 17, 10, 0, tzinfo=timezone.utc),
    )


def main() -> None:
    builder = build_demo()
    result = builder.build(
        user_query="如何解释第九章的上下文工程流程？",
        system_instructions="你是一个教学型 Agent，回答必须基于提供的上下文，不要编造。",
        conversation_history=[
            {
                "role": "user",
                "content": "上一章我们实现了 Memory + RAG。",
                "timestamp": datetime(2026, 6, 17, 9, 30, tzinfo=timezone.utc),
            },
            {
                "role": "assistant",
                "content": "第八章重点是记忆、检索、多模态和 trace。",
                "timestamp": datetime(2026, 6, 17, 9, 40, tzinfo=timezone.utc),
            },
        ],
        custom_packets=[
            ContextPacket(
                source="chapter9",
                content="GSSC 表示 Gather、Select、Structure、Compress，是上下文工程的核心流水线。",
                timestamp=datetime(2026, 6, 17, 9, 55, tzinfo=timezone.utc),
                relevance_score=0.95,
            ),
            ContextPacket(
                source="chapter9",
                content="上下文工程要控制 token 预算，避免把所有历史和证据无差别塞进 prompt。",
                timestamp=datetime(2026, 6, 17, 9, 50, tzinfo=timezone.utc),
                relevance_score=0.9,
            ),
            ContextPacket(
                source="noise",
                content="这是一条和当前问题不相关的低价值信息，会被选择阶段过滤或降级。",
                timestamp=datetime(2026, 6, 16, 9, 0, tzinfo=timezone.utc),
                relevance_score=0.01,
            ),
            ContextPacket(
                source="chapter9",
                content="压缩阶段可以保留高优先级系统指令和高相关证据，同时裁剪低价值长文本。" * 8,
                timestamp=datetime(2026, 6, 17, 9, 45, tzinfo=timezone.utc),
                relevance_score=0.8,
            ),
        ],
        output_instructions="用简洁中文解释，并说明 trace 如何验证。",
    )

    print("[context]")
    print(result.context)
    print("\n[quality]")
    print(
        json.dumps(
            {
                "information_density": round(result.quality.information_density, 3),
                "relevance": round(result.quality.relevance, 3),
                "completeness": round(result.quality.completeness, 3),
                "overall_score": round(result.quality.overall_score, 3),
                "suggestions": result.quality.suggestions,
                "details": result.quality.details,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print("\n[trace]")
    print(json.dumps(result.trace, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
