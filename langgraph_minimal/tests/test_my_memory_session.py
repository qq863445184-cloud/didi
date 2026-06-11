from app.my_memory_system import MyMemoryManager, MyMemoryTool, WorkingMemoryStore


def test_memory_search_can_filter_by_session_id():
    manager = MyMemoryManager(stores={"working": WorkingMemoryStore()})
    tool = MyMemoryTool(manager=manager)

    tool.run(
        {
            "action": "add",
            "content": "会话A：用户正在学习 RAG",
            "session_id": "session-a",
        }
    )
    tool.run(
        {
            "action": "add",
            "content": "会话B：用户正在学习 AgentScope",
            "session_id": "session-b",
        }
    )

    result = tool.run(
        {
            "action": "search",
            "query": "用户 正在学习",
            "session_id": "session-a",
            "limit": 5,
        }
    )

    assert "会话A" in result
    assert "会话B" not in result
    assert tool.trace_events[-1]["session_id"] == "session-a"


def test_memory_summary_can_filter_by_session_id():
    manager = MyMemoryManager(stores={"working": WorkingMemoryStore()})
    tool = MyMemoryTool(manager=manager)

    tool.run({"action": "add", "content": "会话A记忆", "session_id": "a"})
    tool.run({"action": "add", "content": "会话B记忆", "session_id": "b"})

    result = tool.run({"action": "summary", "session_id": "b"})

    assert "记忆总数: 1" in result
    assert "会话B记忆" in result
    assert "会话A记忆" not in result
