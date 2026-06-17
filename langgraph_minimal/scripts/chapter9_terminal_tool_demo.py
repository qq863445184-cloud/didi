from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.context_engineering import ContextBuilder, ContextConfig, ContextPacket
from app.terminal_tool import TerminalTool


def main() -> None:
    demo_dir = ROOT / "memory_data" / "chapter9_terminal_demo"
    demo_dir.mkdir(parents=True, exist_ok=True)
    demo_file = demo_dir / "terminal_context_note.md"
    demo_file.write_text(
        "# TerminalTool\n\n"
        "TerminalTool provides read-only filesystem context for agents.\n"
        "It is useful when a ContextAwareAgent needs live project evidence.\n",
        encoding="utf-8",
    )

    tool = TerminalTool(root=ROOT)

    print("[pwd]")
    print(tool.run({"action": "pwd"}))

    print("\n[list]")
    print(tool.run({"action": "list", "path": "memory_data/chapter9_terminal_demo"}))

    print("\n[read]")
    read_result = tool.run(
        {
            "action": "read",
            "path": "memory_data/chapter9_terminal_demo/terminal_context_note.md",
        }
    )
    print(read_result)

    print("\n[search]")
    print(
        tool.run(
            {
                "action": "search",
                "path": "memory_data/chapter9_terminal_demo",
                "query": "ContextAwareAgent",
                "pattern": "*.md",
            }
        )
    )

    builder = ContextBuilder(config=ContextConfig(max_tokens=700))
    result = builder.build(
        user_query="TerminalTool 在上下文工程里负责什么？",
        system_instructions="你是第九章上下文工程教学助手，必须基于文件证据回答。",
        custom_packets=[
            ContextPacket(
                content=read_result,
                source="terminal",
                relevance_score=0.9,
                metadata={"section": "evidence", "type": "filesystem"},
            )
        ],
        output_instructions="说明它和 NoteTool、RAGTool 的区别。",
    )

    print("\n[context]")
    print(result.context)

    print("\n[terminal trace]")
    print(json.dumps(tool.trace_events, ensure_ascii=False, indent=2))

    print("\n[context trace]")
    print(json.dumps(result.trace, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
