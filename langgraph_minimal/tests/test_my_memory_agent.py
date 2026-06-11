from app.my_memory_agent import MyMemoryAgent


class FakeLLM:
    provider = "fake"

    def __init__(self) -> None:
        self.calls = []

    def invoke(self, messages, **kwargs):
        self.calls.append(messages)
        content = messages[-1]["content"]
        if "工具执行结果" in content and "saved:" in content:
            return "我已经把这条信息保存到记忆里了。"
        if "工具执行结果" in content and "找到 1 条相关记忆" in content:
            return "你叫李雷，是后端开发者。"
        if "请记住" in content:
            return (
                '<tool_call>{"name":"memory","parameters":'
                '{"action":"add","content":"我叫李雷，是后端开发者"}}</tool_call>'
            )
        if "我之前说" in content:
            return (
                '<tool_call>{"name":"memory","parameters":'
                '{"action":"search","query":"我之前说我叫什么，我是什么开发者？"}}</tool_call>'
            )
        return "普通回答。"

    def stream_invoke(self, messages, **kwargs):
        self.calls.append(messages)
        yield "流式"
        yield "回答"


class FakeMemoryTool:
    def __init__(self) -> None:
        self.calls = []

    def to_dict(self):
        return {
            "name": "memory",
            "description": "记忆工具",
            "parameters": [
                {"name": "action", "type": "string", "required": True},
                {"name": "content", "type": "string", "required": False},
                {"name": "query", "type": "string", "required": False},
            ],
        }

    def validate_parameters(self, parameters):
        return "action" in parameters

    def run(self, parameters):
        self.calls.append(parameters)
        action = parameters["action"]
        if action == "add":
            return f"saved:{parameters['content']}"
        if action == "search":
            return "找到 1 条相关记忆：用户叫李雷，是后端开发者。"
        return "unknown"


def test_memory_agent_executes_model_emitted_add_tool_call():
    llm = FakeLLM()
    memory_tool = FakeMemoryTool()
    agent = MyMemoryAgent(name="记忆助手", llm=llm, memory_tool=memory_tool)

    response = agent.run("请记住我叫李雷，是后端开发者")

    assert response == "我已经把这条信息保存到记忆里了。"
    assert agent.last_tool_call["name"] == "memory"
    assert memory_tool.calls[0]["action"] == "add"
    assert memory_tool.calls[0]["content"] == "我叫李雷，是后端开发者"
    assert memory_tool.calls[0]["memory_type"] == "working"
    assert memory_tool.calls[0]["importance"] == 0.9
    assert agent.last_saved_memory_result == "saved:我叫李雷，是后端开发者"
    assert len(llm.calls) == 2
    assert len(agent.get_history()) == 2


def test_memory_agent_executes_model_emitted_search_tool_call():
    llm = FakeLLM()
    memory_tool = FakeMemoryTool()
    agent = MyMemoryAgent(name="记忆助手", llm=llm, memory_tool=memory_tool)

    response = agent.run("我之前说我叫什么，我是什么开发者？")

    assert response == "你叫李雷，是后端开发者。"
    assert memory_tool.calls[0]["action"] == "search"
    assert memory_tool.calls[0]["query"] == "我之前说我叫什么，我是什么开发者？"
    assert memory_tool.calls[0]["limit"] == 5
    assert agent.last_retrieved_memories == "找到 1 条相关记忆：用户叫李雷，是后端开发者。"
    assert len(llm.calls) == 2


def test_memory_agent_does_not_call_memory_without_model_tool_call():
    llm = FakeLLM()
    memory_tool = FakeMemoryTool()
    agent = MyMemoryAgent(name="记忆助手", llm=llm, memory_tool=memory_tool)

    response = agent.run("解释一下 RAG 是什么")

    assert response == "普通回答。"
    assert memory_tool.calls == []
    assert agent.last_tool_call is None
    assert len(llm.calls) == 1


def test_memory_agent_stream_run_delegates_to_tool_call_run():
    llm = FakeLLM()
    memory_tool = FakeMemoryTool()
    agent = MyMemoryAgent(name="记忆助手", llm=llm, memory_tool=memory_tool)

    response = "".join(agent.stream_run("请记住我叫李雷，是后端开发者"))

    assert response == "我已经把这条信息保存到记忆里了。"
    assert memory_tool.calls[0]["action"] == "add"
