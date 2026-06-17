from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.codebase_maintenance_assistant import CodebaseMaintenanceAssistant


class DemoLLM:
    """Tiny deterministic LLM stand-in so the demo is stable offline."""

    def invoke(self, messages, **kwargs):
        return (
            "建议先围绕 app/service.py 的 TODO 建立小步重构计划："
            "抽取 validate_user_id、fetch_user、format_user_response，并把任务继续写入笔记。"
        )


def main() -> None:
    demo_root = ROOT / "memory_data" / "chapter9_codebase_maintenance_demo"
    app_dir = demo_root / "app"
    app_dir.mkdir(parents=True, exist_ok=True)
    (demo_root / "README.md").write_text(
        "# Demo Maintained Service\n\n"
        "A tiny project used to demonstrate long-running codebase maintenance.\n",
        encoding="utf-8",
    )
    (app_dir / "service.py").write_text(
        "def load_user(user_id):\n"
        "    # TODO: split validation, database access, and response formatting\n"
        "    if not user_id:\n"
        "        raise ValueError('missing user_id')\n"
        "    return {'id': user_id, 'name': 'demo'}\n",
        encoding="utf-8",
    )
    note_path = demo_root / "maintenance_notes.jsonl"
    note_path.unlink(missing_ok=True)

    assistant = CodebaseMaintenanceAssistant(
        root=demo_root,
        note_path=note_path,
        context_max_tokens=900,
        llm=DemoLLM(),
    )

    explore = assistant.explore([".", "app"])
    issue = assistant.record_issue(
        title="load_user responsibilities are mixed",
        content="app/service.py 的 load_user 同时包含校验、数据读取和返回格式整理。",
        path="app/service.py",
        importance=0.9,
    )
    task = assistant.track_refactor_task(
        title="Split load_user into smaller functions",
        content="先抽取 validate_user_id，再拆出 fetch_user 和 format_user_response。",
        path="app/service.py",
        status="todo",
        importance=0.85,
    )
    context = assistant.build_maintenance_context(
        "下一步如何维护 app/service.py，并保证长期重构任务不断线？"
    )
    answer = assistant.run(
        "分析 app/service.py 的维护风险，并给出下一步计划",
        mode="analyze",
    )
    second_answer = assistant.run(
        "继续推进 load_user 拆分，并说明今天要保持哪些上下文",
        mode="plan",
    )
    stats = assistant.get_stats()
    report = assistant.generate_report()

    print("[explore.structure]")
    print(explore.structure)
    print("\n[explore.todos]")
    print(explore.todos)
    print("\n[issue note]")
    print(issue)
    print("\n[refactor task]")
    print(task)
    print("\n[maintenance context]")
    print(context.context)
    print("\n[run analyze answer]")
    print(answer)
    print("\n[run plan answer]")
    print(second_answer)
    print("\n[stats]")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print("\n[report]")
    print(report)
    print("\n[trace]")
    print(
        json.dumps(
            {
                "explore": explore.trace,
                "context": context.trace,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
