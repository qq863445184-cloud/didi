from __future__ import annotations

import json
import re
from collections.abc import Iterator
from contextlib import redirect_stdout
from io import StringIO
from typing import Any

from hello_agents import Message, SimpleAgent


class MyMultimodalMemoryAgent(SimpleAgent):
    """让模型自主选择 perception / rag 工具的多模态记忆 Agent。

    这个类只做 Agent 框架层的“机械工作”：
    1. 把可用工具 schema 放进上下文；
    2. 解析模型输出的结构化工具调用；
    3. 执行真实工具并把 observation 回填给模型。

    是否需要调用工具、调用哪个工具，仍然由模型通过 `<tool_call>` 明确表达。
    """

    TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</(?:tool_call|tool_answer)>", re.S)

    def __init__(
        self,
        *args: Any,
        perception_tool: Any,
        rag_tool: Any,
        default_rag_namespace: str = "multimodal",
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.perception_tool = perception_tool
        self.rag_tool = rag_tool
        self.default_rag_namespace = default_rag_namespace
        self.tools = {
            "perception": perception_tool,
            "rag": rag_tool,
        }
        self.last_model_messages: list[dict[str, str]] = []
        self.last_tool_call: dict[str, Any] | None = None
        self.last_tool_result: str | None = None
        self.trace_events: list[dict[str, Any]] = []

    def run(self, input_text: str, **kwargs: Any) -> str:
        trace = bool(kwargs.pop("trace", False))
        max_tool_iterations = int(kwargs.pop("max_tool_iterations", 4))
        self.last_tool_call = None
        self.last_tool_result = None
        self.trace_events = []

        messages = self._build_messages(input_text)
        self._record_model_input(messages, trace=trace, label="MultimodalMemoryAgent LLM 输入 #1")
        response = self.llm.invoke(messages, **kwargs)

        # 多轮工具循环：模型可以先 ingest 文件，再 search RAG，最后自然语言回答。
        # 框架不猜意图，只要模型继续给出合法工具调用，就继续执行。
        for iteration in range(max_tool_iterations):
            tool_call = self._parse_tool_call(response)
            if tool_call is None:
                break

            self.last_tool_call = tool_call
            tool_result = self._execute_tool(tool_call)
            self.last_tool_result = tool_result

            messages = self._build_messages(
                input_text,
                previous_messages=messages,
                tool_call=response,
                tool_result=tool_result,
            )
            self._record_model_input(
                messages,
                trace=trace,
                label=f"MultimodalMemoryAgent LLM 输入 #{iteration + 2}",
            )
            response = self.llm.invoke(messages, **kwargs)

        self.add_message(Message(input_text, "user"))
        self.add_message(Message(response, "assistant"))
        return response

    def stream_run(self, input_text: str, **kwargs: Any) -> Iterator[str]:
        # 先复用 run，保证流式和非流式的工具执行语义完全一致。
        # 后续如果接入真正 token streaming，可以只在最终回答阶段流式吐出。
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
            system_prompt = self.system_prompt or "你是一个具备多模态记忆和检索能力的中文助手。"
            messages.append(
                {
                    "role": "system",
                    "content": "\n\n".join([system_prompt, self._tool_prompt()]),
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
                        "如果还需要使用 perception 或 rag 工具，可以继续输出工具调用；",
                        "如果已经足够，请直接回答用户，不要再输出工具调用。",
                    ]
                ),
            }
        )
        return messages

    def _tool_prompt(self) -> str:
        perception_schema = self.perception_tool.to_dict()
        rag_schema = self.rag_tool.to_dict()
        return "\n".join(
            [
                "你可以使用两个工具：",
                "1. perception：处理图片、音频、视频、文档等多模态文件，并写入感知记忆或知识库。",
                "2. rag：检索已经入库的文本、转写内容、图片描述或多模态索引结果。",
                "只有当确实需要工具时，才输出工具调用；普通问答直接回答。",
                "如果需要调用工具，请只输出如下格式，不要解释：",
                '<tool_call>{"name":"perception","parameters":{"action":"ingest_file","file_path":"文件路径","description":"可选说明"}}</tool_call>',
                f'<tool_call>{{"name":"rag","parameters":{{"action":"search","query":"检索问题","namespace":"{self.default_rag_namespace}"}}}}</tool_call>',
                "rag 工具如果缺少 namespace，框架会默认补为 multimodal。",
                "perception 工具 schema：",
                json.dumps(perception_schema, ensure_ascii=False),
                "rag 工具 schema：",
                json.dumps(rag_schema, ensure_ascii=False),
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

        name = payload.get("name")
        if name not in self.tools:
            return None
        parameters = payload.get("parameters", {})
        if not isinstance(parameters, dict):
            return None

        # RAG 一般会按 namespace 隔离不同知识库。模型漏填时补默认值，
        # 但“是否调用 rag”仍然必须来自模型显式输出。
        if name == "rag":
            parameters.setdefault("namespace", self.default_rag_namespace)
        return {"name": name, "parameters": parameters}

    def _execute_tool(self, tool_call: dict[str, Any]) -> str:
        name = tool_call["name"]
        parameters = tool_call["parameters"]
        tool = self.tools[name]

        if hasattr(tool, "validate_parameters") and not tool.validate_parameters(parameters):
            result = f"错误：{name} 工具参数不完整"
        else:
            # 一些底层工具会打印进度或模型加载日志。这里捕获 stdout，
            # 让 Agent 的 observation 只保留工具返回值，trace 更干净。
            with redirect_stdout(StringIO()):
                result = tool.run(parameters)

        self.trace_events.append(
            {"stage": "tool", "tool": name, "parameters": parameters, "result": result}
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
