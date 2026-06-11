from app.my_memory_system import MyMemoryManager, MyMemoryTool, WorkingMemoryStore


def _tool_with_two_local_stores() -> MyMemoryTool:
    manager = MyMemoryManager(
        stores={
            "working": WorkingMemoryStore(),
            "semantic": WorkingMemoryStore(),
        }
    )
    return MyMemoryTool(manager=manager)


def test_memory_tool_stats_reports_counts_by_type():
    tool = _tool_with_two_local_stores()
    tool.run({"action": "add", "content": "短期任务：整理第八章", "memory_type": "working"})
    tool.run({"action": "add", "content": "长期事实：用户在学习 Agent", "memory_type": "semantic"})

    result = tool.run({"action": "stats"})

    assert "记忆统计" in result
    assert "working: 1" in result
    assert "semantic: 1" in result
    assert "total: 2" in result


def test_memory_tool_update_changes_existing_record():
    tool = _tool_with_two_local_stores()
    tool.run({"action": "add", "content": "用户是后端开发者", "memory_type": "working"})
    record = tool.manager.stores["working"].records[0]

    result = tool.run(
        {
            "action": "update",
            "memory_type": "working",
            "memory_id": record.memory_id,
            "content": "用户是后端开发者，正在学习 RAG",
            "importance": 0.95,
            "topic": "rag",
        }
    )

    assert "已更新记忆" in result
    assert tool.manager.stores["working"].records[0].content.endswith("学习 RAG")
    assert tool.manager.stores["working"].records[0].importance == 0.95
    assert tool.manager.stores["working"].records[0].metadata["topic"] == "rag"


def test_memory_tool_remove_deletes_one_record():
    tool = _tool_with_two_local_stores()
    tool.run({"action": "add", "content": "可以删除的短期记忆", "memory_type": "working"})
    record = tool.manager.stores["working"].records[0]

    result = tool.run(
        {
            "action": "remove",
            "memory_type": "working",
            "memory_id": record.memory_id,
        }
    )

    assert "已删除记忆" in result
    assert tool.manager.stores["working"].records == []


def test_memory_tool_clear_all_can_clear_one_type_or_all_types():
    tool = _tool_with_two_local_stores()
    tool.run({"action": "add", "content": "工作记忆", "memory_type": "working"})
    tool.run({"action": "add", "content": "语义记忆", "memory_type": "semantic"})

    one_type = tool.run({"action": "clear_all", "memory_type": "working"})
    all_types = tool.run({"action": "clear_all", "memory_type": "all"})

    assert "working: 1" in one_type
    assert "semantic: 1" in all_types
    assert tool.manager.stores["working"].records == []
    assert tool.manager.stores["semantic"].records == []


def test_memory_tool_searches_across_memory_types_and_filters_importance():
    tool = _tool_with_two_local_stores()
    tool.run(
        {
            "action": "add",
            "content": "工作记忆：用户正在学习 Agent 记忆系统",
            "memory_type": "working",
            "importance": 0.4,
        }
    )
    tool.run(
        {
            "action": "add",
            "content": "语义记忆：Agent 记忆系统包含检索增强",
            "memory_type": "semantic",
            "importance": 0.9,
        }
    )

    result = tool.run(
        {
            "action": "search",
            "query": "Agent 记忆系统",
            "memory_type": "all",
            "min_importance": 0.8,
            "limit": 5,
        }
    )

    assert "找到 1 条相关记忆" in result
    assert "语义记忆" in result
    assert "工作记忆" not in result


def test_working_memory_auto_prunes_by_ttl_and_capacity():
    store = WorkingMemoryStore(ttl_seconds=10, max_records=2)
    old = store.add_record(content="过期记忆", importance=0.9)
    old.created_at = 100.0
    store.add_record(content="重要记忆", importance=0.9)
    store.add_record(content="普通记忆", importance=0.5)

    store.prune(now=111.0)

    assert [record.content for record in store.records] == ["重要记忆", "普通记忆"]

    store.add_record(content="最新低价值记忆", importance=0.1)

    assert len(store.records) == 2
    assert "重要记忆" in {record.content for record in store.records}
