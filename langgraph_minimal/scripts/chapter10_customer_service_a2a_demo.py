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

from app.chapter10_customer_service_a2a import (
    create_customer_service_agents,
    handle_customer_request,
    list_customer_service_agent_cards,
)


def main() -> None:
    agents = create_customer_service_agents()

    print("[customer service agent cards]")
    print(json.dumps(list_customer_service_agent_cards(agents), ensure_ascii=False, indent=2))

    questions = [
        "我登录一直报错，提示 token 无效，应该怎么办？",
        "你们有没有试用套餐，后续怎么购买？",
        "我这个问题比较复杂，想找人工客服处理。",
    ]

    for question in questions:
        print("\n[customer request]")
        print(question)
        print(json.dumps(handle_customer_request(question, agents), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
