from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from hello_agents import Message, SimpleAgent

from app.context_engineering import (
    ContextBuilder,
    ContextBuildResult,
    ContextConfig,
    ContextPacket,
)


class MyContextAwareAgent(SimpleAgent):
    """Agent that answers through an explicit context-engineering pipeline.

    第九章的重点是：不要把所有历史、记忆、RAG 证据直接拼给模型，而是
    先构建可控上下文。这个 Agent 在每次调用 LLM 前都会运行
    ContextBuilder，并把最终 context 作为用户消息交给模型。
    """

    def __init__(
        self,
        *args: Any,
        context_builder: ContextBuilder | None = None,
        context_config: ContextConfig | None = None,
        memory_tool: Any | None = None,
        rag_tool: Any | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.context_builder = context_builder or ContextBuilder(
            config=context_config,
            memory_tool=memory_tool,
            rag_tool=rag_tool,
        )
        self.last_context_result: ContextBuildResult | None = None
        self.last_model_messages: list[dict[str, str]] = []
        self.trace_events: list[dict[str, Any]] = []

    def run(self, input_text: str, **kwargs: Any) -> str:
        trace = bool(kwargs.pop("trace", False))
        custom_packets = kwargs.pop("custom_packets", None)
        output_instructions = str(
            kwargs.pop("output_instructions", "请基于上下文给出简洁、准确的中文回答。")
        )

        context_result = self._build_context(
            input_text=input_text,
            custom_packets=custom_packets or [],
            output_instructions=output_instructions,
        )
        self.last_context_result = context_result
        self.trace_events = list(context_result.trace)

        messages = self._build_messages_from_context(context_result.context)
        self._record_model_input(messages, trace=trace)
        response = self.llm.invoke(messages, **kwargs)

        self.add_message(Message(input_text, "user"))
        self.add_message(Message(response, "assistant"))
        self.trace_events.append(
            {
                "stage": "agent.answer",
                "input": input_text,
                "context_tokens": context_result.total_tokens,
                "history_messages": len(self.get_history()),
            }
        )
        return response

    def stream_run(self, input_text: str, **kwargs: Any) -> Iterator[str]:
        trace = bool(kwargs.pop("trace", False))
        custom_packets = kwargs.pop("custom_packets", None)
        output_instructions = str(
            kwargs.pop("output_instructions", "请基于上下文给出简洁、准确的中文回答。")
        )

        context_result = self._build_context(
            input_text=input_text,
            custom_packets=custom_packets or [],
            output_instructions=output_instructions,
        )
        self.last_context_result = context_result
        self.trace_events = list(context_result.trace)

        messages = self._build_messages_from_context(context_result.context)
        self._record_model_input(messages, trace=trace)

        response = ""
        for chunk in self.llm.stream_invoke(messages, **kwargs):
            response += chunk
            yield chunk

        self.add_message(Message(input_text, "user"))
        self.add_message(Message(response, "assistant"))
        self.trace_events.append(
            {
                "stage": "agent.answer_stream",
                "input": input_text,
                "context_tokens": context_result.total_tokens,
                "history_messages": len(self.get_history()),
            }
        )

    def _build_context(
        self,
        *,
        input_text: str,
        custom_packets: list[ContextPacket],
        output_instructions: str,
    ) -> ContextBuildResult:
        history_packets = [
            {"role": message.role, "content": message.content}
            for message in self.get_history()
        ]
        return self.context_builder.build(
            user_query=input_text,
            system_instructions=self.system_prompt or "",
            conversation_history=history_packets,
            custom_packets=custom_packets,
            output_instructions=output_instructions,
        )

    def _build_messages_from_context(self, context: str) -> list[dict[str, str]]:
        # system_prompt 已经被结构化进 context 的 Role & Policies 区块；
        # 这里用单条 user message，便于测试和 trace 直接看到完整上下文包。
        return [{"role": "user", "content": context}]

    def _record_model_input(
        self,
        messages: list[dict[str, str]],
        *,
        trace: bool,
    ) -> None:
        self.last_model_messages = messages
        if trace:
            print("\n--- ContextAwareAgent LLM 输入 ---")
            print(json.dumps(messages, ensure_ascii=False, indent=2))
