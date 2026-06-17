from __future__ import annotations

from app.codebase_maintenance_assistant import CodebaseMaintenanceAssistant


def test_codebase_maintenance_assistant_explores_records_tasks_and_builds_context(tmp_path):
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    (app_dir / "service.py").write_text(
        "def load_user(user_id):\n"
        "    # TODO: split database access from validation\n"
        "    return {'id': user_id}\n",
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text(
        "# Demo Service\n\nSmall codebase for maintenance assistant tests.\n",
        encoding="utf-8",
    )

    assistant = CodebaseMaintenanceAssistant(
        root=tmp_path,
        note_path=tmp_path / "maintenance_notes.jsonl",
        context_max_tokens=420,
    )

    explore = assistant.explore([".", "app"])
    assert "README.md" in explore.structure
    assert "service.py" in explore.structure

    issue = assistant.record_issue(
        title="service.py mixes responsibilities",
        content="app/service.py 的 load_user 同时处理数据访问和校验，后续应拆分。",
        path="app/service.py",
        importance=0.9,
    )
    assert "已保存笔记" in issue

    task = assistant.track_refactor_task(
        title="Split load_user responsibilities",
        content="将 app/service.py 的数据访问、校验和返回格式整理拆成独立函数。",
        path="app/service.py",
        status="todo",
        importance=0.8,
    )
    assert "已保存笔记" in task

    result = assistant.build_maintenance_context(
        "如何维护 load_user 并保持后续重构任务连贯？"
    )

    assert "[Evidence]" in result.context
    assert "[Context]" in result.context
    assert "source=terminal" in result.context
    assert "source=note" in result.context
    assert "source=memory" in result.context
    assert "load_user" in result.context
    assert result.total_tokens <= 420
    assert any(event["stage"] == "maintenance.context" for event in result.trace)


def test_codebase_maintenance_assistant_can_resume_long_refactor_notes(tmp_path):
    (tmp_path / "main.py").write_text("print('hello')\n", encoding="utf-8")
    note_path = tmp_path / "notes.jsonl"

    first = CodebaseMaintenanceAssistant(root=tmp_path, note_path=note_path)
    first.track_refactor_task(
        title="Introduce service layer",
        content="长期任务：把 main.py 中的业务逻辑迁移到 service 层。",
        path="main.py",
        status="in_progress",
    )

    resumed = CodebaseMaintenanceAssistant(root=tmp_path, note_path=note_path)
    context = resumed.build_maintenance_context("继续推进 service layer 重构").context

    assert "Introduce service layer" in context
    assert "长期任务" in context
