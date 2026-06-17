from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.note_tool import NoteTool


def main() -> None:
    note_path = ROOT / "memory_data" / "chapter9_notes_demo.jsonl"
    tool = NoteTool(path=note_path)

    print("[add]")
    print(
        tool.run(
            {
                "action": "add",
                "title": "上下文工程结构",
                "content": "第九章上下文构建采用 Role、Task、Evidence、Context、Output 五段式结构。",
                "tags": ["chapter9", "context"],
                "importance": 0.9,
            }
        )
    )

    print("\n[search]")
    print(tool.run({"action": "search", "query": "Evidence Context Output"}))

    print("\n[stats]")
    print(tool.run({"action": "stats"}))

    print("\n[trace]")
    print(json.dumps(tool.trace_events, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
