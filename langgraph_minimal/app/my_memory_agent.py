from __future__ import annotations

import json
import re
from collections.abc import Iterator
from contextlib import redirect_stdout
from io import StringIO
from typing import Any

from hello_agents import Message, SimpleAgent
from hello_agents.tools import MemoryTool


class MyMemoryAgent(SimpleAgent):
    """A memory-aware teaching agent driven by model tool calls.

    The agent itself does not decide memory intent by keywords. Instead, it
    exposes the MemoryTool schema to the model and asks the model to emit a
    tool call when memory is needed. Python only does the mechanical framework
    work: parse the call, execute MemoryTool, and send the observation back.
    """

    TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</(?:tool_call|tool_answer)>", re.S)

    def __init__(
        self,
        *args: Any,
        memory_tool: MemoryTool | None = None,
        memory_type: str = "working",
        recall_limit: int = 5,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.memory_tool = memory_tool or MemoryTool(memory_types=[memory_type])
        self.memory_type = memory_type
        self.recall_limit = recall_limit
        self.last_model_messages: list[dict[str, str]] = []
        self.last_tool_call: dict[str, Any] | None = None
        self.last_tool_result: str | None = None
        self.last_saved_memory_result: str | None = None
        self.last_retrieved_memories: str | None = None
        self.trace_events: list[dict[str, Any]] = []

    def run(self, input_text: str, **kwargs: Any) -> str:
        trace = bool(kwargs.pop("trace", False))
        max_tool_iterations = int(kwargs.pop("max_tool_iterations", 3))
        self.last_tool_call = None
        self.last_tool_result = None
        self.last_saved_memory_result = None
        self.last_retrieved_memories = None
        self.trace_events = []

        messages = self._build_messages(input_text)
        self._record_model_input(messages, trace=trace, label="MemoryAgent LLM 输入 #1")
        response = self.llm.invoke(messages, **kwargs)

        # 只要模型继续输出 memory 工具调用，框架就继续执行并把结果回填。
        # 这里不判断“用户是不是想记忆”，只响应模型已经明确给出的工具调用。
        for iteration in range(max_tool_iterations):
            tool_call = self._parse_tool_call(response)
            if tool_call is None:
                break

            self.last_tool_call = tool_call
            tool_result = self._execute_memory_tool(tool_call)
            self.last_tool_result = tool_result

            action = tool_call["parameters"].get("action")
            if action == "add":
                self.last_saved_memory_result = tool_result
            elif action == "search":
                self.last_retrieved_memories = tool_result

            messages = self._build_messages(
                input_text,
                previous_messages=messages,
                tool_call=response,
                tool_result=tool_result,
            )
            self._record_model_input(
                messages,
                trace=trace,
                label=f"MemoryAgent LLM 输入 #{iteration + 2}",
            )
            response = self.llm.invoke(messages, **kwargs)

        self.add_message(Message(input_text, "user"))
        self.add_message(Message(response, "assistant"))
        return response

    def stream_run(self, input_text: str, **kwargs: Any) -> Iterator[str]:
        # 这里复用 run，是为了保证“模型决定工具调用 -> 框架执行工具”
        # 这条路径完全一致。后续要做 token streaming 时，可在无工具调用
        # 的第二次 LLM 回复阶段再接入 stream_invoke。
        yield self.run(input_text, **kwargs)

    def _build_messages(
        self,
        input_text: str,
        *,
        previous_messages: list[dict[str, str]] | None = None,
        tool_call: str | None = None,
        tool_result: str | None = None,
    ) -> list[dict[str, str]]:
        if previous_messages is None:
            messages: list[dict[str, str]] = []
            system_prompt = self.system_prompt or "你是一个有记忆能力的中文助手。"
            messages.append(
                {
                    "role": "system",
                    "content": "\n\n".join([system_prompt, self._memory_tool_prompt()]),
                }
            )

            for msg in self._history:
                messages.append({"role": msg.role, "content": msg.content})
            messages.append({"role": "user", "content": input_text})
            return messages

        messages = list(previous_messages)
        if tool_call:
            messages.append({"role": "assistant", "content": tool_call})
        messages.append(
            {
                "role": "user",
                "content": "\n".join(
                    [
                        f"工具执行结果：{tool_result}",
                        "如果还需要使用 memory 工具，可以继续输出工具调用；",
                        "如果已经足够，请直接回答用户，不要再输出工具调用。",
                    ]
                ),
            }
        )
        return messages

    def _memory_tool_prompt(self) -> str:
        tool_schema = self.memory_tool.to_dict()
        default_params = {
            "memory_type": self.memory_type,
            "limit": self.recall_limit,
        }
        return "\n".join(
            [
                "你可以使用 memory 工具来存储和检索用户记忆。",
                "当用户要求你记住、保存某个事实或偏好时，调用 memory 工具执行 add。",
                "当用户询问之前说过什么、个人信息或偏好时，调用 memory 工具执行 search。",
                "搜索记忆时，query 必须使用短关键词或实体词，不要直接使用“我叫什么”这类完整问句。",
                "例如用户问“我之前说我叫什么？我是什么开发者？”，应搜索“王小明”或“后端开发者”；如果不知道具体名字，就搜索“用户 后端开发者”。",
                "如果需要调用工具，请只输出如下格式，不要解释：",
                '<tool_call>{"name":"memory","parameters":{"action":"add","content":"要保存的内容","memory_type":"working","importance":0.9}}</tool_call>',
                '<tool_call>{"name":"memory","parameters":{"action":"search","query":"要检索的问题","memory_type":"working","limit":5}}</tool_call>',
                f"默认参数：{json.dumps(default_params, ensure_ascii=False)}",
                "memory 工具 schema：",
                json.dumps(tool_schema, ensure_ascii=False),
            ]
        )

    def _parse_tool_call(self, text: str) -> dict[str, Any] | None:
        match = self.TOOL_CALL_RE.search(text)
        if not match:
            return None
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        if payload.get("name") != "memory":
            return None
        parameters = payload.get("parameters", {})
        if not isinstance(parameters, dict):
            return None

        # 给模型省一点负担：如果它漏了默认参数，框架补默认值；
        # 但“是否调用 memory”这个决策仍然完全来自模型输出。
        action = parameters.get("action")
        parameters.setdefault("memory_type", self.memory_type)
        if action == "add":
            parameters.setdefault("importance", 0.9)
        elif action == "search":
            parameters.setdefault("limit", self.recall_limit)
        return {"name": "memory", "parameters": parameters}

    def _execute_memory_tool(self, tool_call: dict[str, Any]) -> str:
        parameters = tool_call["parameters"]
        if hasattr(self.memory_tool, "validate_parameters") and not self.memory_tool.validate_parameters(
            parameters
        ):
            result = "错误：memory 工具参数不完整"
        else:
            with redirect_stdout(StringIO()):
                result = self.memory_tool.run(parameters)

        self.trace_events.append(
            {"stage": "memory_tool", "parameters": parameters, "result": result}
        )
        return result

    def _record_model_input(
        self,
        messages: list[dict[str, str]],
        *,
        trace: bool,
        label: str,
    ) -> None:
        self.last_model_messages = messages
        if trace:
            print(f"\n--- {label} ---")
            print(json.dumps(messages, ensure_ascii=False, indent=2))
