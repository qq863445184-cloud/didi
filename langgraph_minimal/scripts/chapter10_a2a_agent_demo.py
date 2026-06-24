from __future__ import annotations

import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.chapter10_a2a_agent import call_a2a_skill, create_simple_a2a_agent


def main() -> None:
    agent = create_simple_a2a_agent()

    print("[agent card]")
    print(json.dumps(agent.get_info(), ensure_ascii=False, indent=2))

    print("\n[call introduce]")
    print(json.dumps(call_a2a_skill(agent, "introduce", ""), ensure_ascii=False, indent=2))

    print("\n[call calculate]")
    print(json.dumps(call_a2a_skill(agent, "calculate", "请计算 2 + 3 * 4"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
