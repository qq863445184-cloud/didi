from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.context_engineering import ContextBuilder, ContextConfig, ContextPacket
from app.note_tool import NoteTool


def main() -> None:
    note_path = ROOT / "memory_data" / "chapter9_note_context_demo.jsonl"
    note_path.unlink(missing_ok=True)
    note_tool = NoteTool(path=note_path)

    note_tool.run(
        {
            "action": "add",
            "title": "五段式上下文",
            "content": "第九章 prompt 推荐组织为 Role & Policies、Task、Evidence、Context、Output。",
            "tags": ["chapter9", "context"],
            "importance": 0.9,
        }
    )
    note_tool.run(
        {
            "action": "add",
            "title": "选择策略",
            "content": "ContextBuilder 会按相关性、新近性和 token 预算选择上下文片段。",
            "tags": ["chapter9", "select"],
            "importance": 0.8,
        }
    )

    note_result = note_tool.run(
        {
            "action": "search",
            "query": "上下文 prompt Evidence Context Output",
            "limit": 3,
        }
    )

    builder = ContextBuilder(config=ContextConfig(max_tokens=700))
    result = builder.build(
        user_query="如何用 NoteTool 辅助构建第九章上下文？",
        system_instructions="你是第九章上下文工程教学助手，必须基于提供的上下文回答。",
        custom_packets=[
            ContextPacket(
                content=note_result,
                source="note",
                relevance_score=0.85,
                metadata={"section": "context", "type": "task_note"},
            )
        ],
        output_instructions="先说明流程，再说明 trace 应该看什么。",
    )

    print("[note search result]")
    print(note_result)
    print("\n[context]")
    print(result.context)
    print("\n[note trace]")
    print(json.dumps(note_tool.trace_events, ensure_ascii=False, indent=2))
    print("\n[context trace]")
    print(json.dumps(result.trace, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
