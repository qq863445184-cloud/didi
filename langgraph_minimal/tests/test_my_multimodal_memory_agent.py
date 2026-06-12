from app.my_multimodal_memory_agent import MyMultimodalMemoryAgent


class FakeLLM:
    provider = "fake"

    def __init__(self) -> None:
        self.calls = []

    def invoke(self, messages, **kwargs):
        self.calls.append(messages)
        content = messages[-1]["content"]
        if "工具执行结果" in content and "Perceptual memory saved" in content:
            return "已经把这个文件写入多模态记忆。"
        if "工具执行结果" in content and "搜索结果" in content:
            return "刚才音频里提到了 ASR、CLAP 和多模态检索。"
        if "请记住这个图片" in content:
            return (
                '<tool_call>{"name":"perception","parameters":'
                '{"action":"ingest_file","file_path":"diagram.png","description":"架构图"}}'
                "</tool_call>"
            )
        if "刚才音频里说了什么" in content:
            return (
                '<tool_call>{"name":"rag","parameters":'
                '{"action":"search","query":"音频 ASR CLAP 多模态检索"}}'
                "</tool_call>"
            )
        return "普通回答。"


class FakeTool:
    def __init__(self, name: str) -> None:
        self.name = name
        self.calls = []

    def to_dict(self):
        return {
            "name": self.name,
            "description": f"{self.name} tool",
            "parameters": [
                {"name": "action", "type": "string", "required": True},
                {"name": "file_path", "type": "string", "required": False},
                {"name": "query", "type": "string", "required": False},
                {"name": "namespace", "type": "string", "required": False},
            ],
        }

    def validate_parameters(self, parameters):
        return "action" in parameters

    def run(self, parameters):
        self.calls.append(parameters)
        if self.name == "perception":
            return f"Perceptual memory saved: {parameters['file_path']}"
        if self.name == "rag":
            return "搜索结果：音频里提到 ASR、CLAP 和多模态检索"
        return "unknown"


def test_multimodal_agent_executes_model_emitted_perception_call():
    llm = FakeLLM()
    perception_tool = FakeTool("perception")
    rag_tool = FakeTool("rag")
    agent = MyMultimodalMemoryAgent(
        name="多模态助手",
        llm=llm,
        perception_tool=perception_tool,
        rag_tool=rag_tool,
    )

    response = agent.run("请记住这个图片 diagram.png")

    assert response == "已经把这个文件写入多模态记忆。"
    assert agent.last_tool_call["name"] == "perception"
    assert perception_tool.calls == [
        {"action": "ingest_file", "file_path": "diagram.png", "description": "架构图"}
    ]
    assert rag_tool.calls == []
    assert agent.last_tool_result == "Perceptual memory saved: diagram.png"
    assert len(llm.calls) == 2
    assert len(agent.get_history()) == 2


def test_multimodal_agent_executes_model_emitted_rag_call():
    llm = FakeLLM()
    perception_tool = FakeTool("perception")
    rag_tool = FakeTool("rag")
    agent = MyMultimodalMemoryAgent(
        name="多模态助手",
        llm=llm,
        perception_tool=perception_tool,
        rag_tool=rag_tool,
    )

    response = agent.run("刚才音频里说了什么？")

    assert response == "刚才音频里提到了 ASR、CLAP 和多模态检索。"
    assert rag_tool.calls == [
        {"action": "search", "query": "音频 ASR CLAP 多模态检索", "namespace": "multimodal"}
    ]
    assert perception_tool.calls == []
    assert agent.last_tool_call["name"] == "rag"
    assert len(llm.calls) == 2


def test_multimodal_agent_does_not_call_tool_without_model_tool_call():
    llm = FakeLLM()
    perception_tool = FakeTool("perception")
    rag_tool = FakeTool("rag")
    agent = MyMultimodalMemoryAgent(
        name="多模态助手",
        llm=llm,
        perception_tool=perception_tool,
        rag_tool=rag_tool,
    )

    response = agent.run("解释一下多模态 RAG")

    assert response == "普通回答。"
    assert perception_tool.calls == []
    assert rag_tool.calls == []
    assert agent.last_tool_call is None
    assert len(llm.calls) == 1


def test_multimodal_agent_prompt_contains_both_tool_schemas():
    llm = FakeLLM()
    perception_tool = FakeTool("perception")
    rag_tool = FakeTool("rag")
    agent = MyMultimodalMemoryAgent(
        name="多模态助手",
        llm=llm,
        perception_tool=perception_tool,
        rag_tool=rag_tool,
    )

    agent.run("解释一下多模态 RAG")

    system_prompt = llm.calls[0][0]["content"]
    assert "perception" in system_prompt
    assert "rag" in system_prompt
    assert "<tool_call>" in system_prompt


def test_multimodal_agent_stream_run_delegates_to_run():
    llm = FakeLLM()
    perception_tool = FakeTool("perception")
    rag_tool = FakeTool("rag")
    agent = MyMultimodalMemoryAgent(
        name="多模态助手",
        llm=llm,
        perception_tool=perception_tool,
        rag_tool=rag_tool,
    )

    response = "".join(agent.stream_run("请记住这个图片 diagram.png"))

    assert response == "已经把这个文件写入多模态记忆。"
    assert perception_tool.calls[0]["action"] == "ingest_file"
