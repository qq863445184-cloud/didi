from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.context_engineering import ContextConfig, ContextPacket
from app.my_context_aware_agent import MyContextAwareAgent


class DemoLLM:
    provider = "demo"

    def invoke(self, messages, **kwargs):
        prompt = messages[-1]["content"]
        if "GSSC" in prompt:
            return (
                "第九章上下文工程可以概括为 GSSC：Gather 收集候选上下文，"
                "Select 按相关性和预算选择，Structure 组织成清晰区块，"
                "Compress 在超预算时压缩低优先级内容。"
            )
        return "已基于上下文回答。"

    def stream_invoke(self, messages, **kwargs):
        yield self.invoke(messages, **kwargs)


def main() -> None:
    agent = MyContextAwareAgent(
        name="第九章上下文工程助手",
        llm=DemoLLM(),
        system_prompt="你是一个教学型 Agent，只能基于上下文工程包回答。",
        context_config=ContextConfig(max_tokens=420, reserve_ratio=0.15),
    )
    answer = agent.run(
        "第九章的上下文工程流程是什么？",
        custom_packets=[
            ContextPacket(
                source="chapter9",
                content="GSSC 是上下文工程的核心流程：Gather、Select、Structure、Compress。",
                relevance_score=0.98,
            ),
            ContextPacket(
                source="chapter9",
                content="上下文工程要区分系统指令、任务、状态、证据和历史，避免无差别拼接。",
                relevance_score=0.9,
            ),
            ContextPacket(
                source="noise",
                content="这条内容和第九章无关，应被过滤。",
                relevance_score=0.01,
            ),
        ],
        trace=True,
    )
    print("\n[answer]")
    print(answer)
    print("\n[context trace]")
    print(json.dumps(agent.trace_events, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
