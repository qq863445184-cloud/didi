from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hello_agents import HelloAgentsLLM

from app.chapter10_customer_service_a2a import create_customer_service_agents, handle_customer_request


def main() -> None:
    load_dotenv(override=True)
    llm = HelloAgentsLLM(
        provider=os.getenv("LLM_PROVIDER", "deepseek"),
        model=os.getenv("MODEL_NAME"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        temperature=0,
        max_tokens=int(os.getenv("A2A_RECEPTION_MAX_TOKENS", "200")),
        timeout=60,
    )
    agents = create_customer_service_agents(llm=llm)

    for question in [
        "我登录一直报错，提示 token 无效，应该怎么办？",
        "你们有没有试用套餐，后续怎么购买？",
        "我这个问题比较复杂，想找人工客服处理。",
    ]:
        result = handle_customer_request(question, agents=agents)
        result["raw_reception_route"] = getattr(agents["reception"], "last_raw_route", "")
        print("\n[real llm customer request]")
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
