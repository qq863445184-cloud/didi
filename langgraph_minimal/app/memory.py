import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MEMORY_ROOT = PROJECT_ROOT / "memory"
SESSIONS_ROOT = MEMORY_ROOT / "sessions"
PROFILE_PATH = MEMORY_ROOT / "profile.json"
PROJECT_MEMORY_PATH = MEMORY_ROOT / "project.json"

DEFAULT_PROFILE = {
    "language": "zh-CN",
    "preferences": [
        "先讲清原理，再实现",
        "不盲目扩功能",
        "优先优化 Agent 架构方案",
    ],
    "defaults": {
        "model": "gpt-5.5",
        "rag_mode": "fast",
    },
}

DEFAULT_PROJECT_MEMORY = {
    "project": "langgraph_minimal",
    "facts": [
        "项目使用 LangGraph 编排 agent 流程",
        "项目使用 LangChain 构建本地 RAG 检索链路",
        "默认 CLI 通过 router 自动分流 general agent 和 agentic RAG",
    ],
    "decisions": [
        "Working Memory 保留在 LangGraph State 中，不落盘",
        "Conversation Memory、Profile Memory、Project Memory 落盘",
        "默认 RAG 使用 fast mode，保留 sentence_transformers 作为高质量模式",
    ],
    "issues": [
        "OpenAI-compatible endpoint 当前没有可用 text-embedding-3-small embedding 通道",
        "sentence-transformers 单次 CLI 冷启动较慢",
    ],
}

DEFAULT_SESSION = {
    "summary": "",
    "recent_messages": [],
}


def ensure_memory_files() -> None:
    MEMORY_ROOT.mkdir(exist_ok=True)
    SESSIONS_ROOT.mkdir(exist_ok=True)
    if not PROFILE_PATH.exists():
        _write_json(PROFILE_PATH, DEFAULT_PROFILE)
    if not PROJECT_MEMORY_PATH.exists():
        _write_json(PROJECT_MEMORY_PATH, DEFAULT_PROJECT_MEMORY)


def load_profile() -> dict[str, Any]:
    ensure_memory_files()
    return _read_json(PROFILE_PATH, DEFAULT_PROFILE)


def load_project_memory() -> dict[str, Any]:
    ensure_memory_files()
    return _read_json(PROJECT_MEMORY_PATH, DEFAULT_PROJECT_MEMORY)


def load_session(session_id: str = "default") -> dict[str, Any]:
    ensure_memory_files()
    path = _session_path(session_id)
    if not path.exists():
        _write_json(path, DEFAULT_SESSION)
    return _read_json(path, DEFAULT_SESSION)


def save_session(session: dict[str, Any], session_id: str = "default") -> None:
    ensure_memory_files()
    _write_json(_session_path(session_id), session)


def append_session_turn(user: str, assistant: str, session_id: str = "default") -> None:
    session = load_session(session_id)
    recent = session.setdefault("recent_messages", [])
    recent.extend(
        [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    )

    max_messages = 10
    if len(recent) > max_messages:
        overflow = recent[:-max_messages]
        session["recent_messages"] = recent[-max_messages:]
        session["summary"] = _append_summary(session.get("summary", ""), overflow)

    save_session(session, session_id)


def get_memory_context(session_id: str = "default") -> str:
    profile = load_profile()
    project = load_project_memory()
    session = load_session(session_id)

    lines = ["Memory context:"]
    lines.append(f"- User language: {profile.get('language', 'zh-CN')}")
    lines.extend(f"- Preference: {item}" for item in profile.get("preferences", []))
    for key, value in profile.get("defaults", {}).items():
        lines.append(f"- Default {key}: {value}")

    lines.append(f"- Project: {project.get('project', 'unknown')}")
    lines.extend(f"- Project fact: {item}" for item in project.get("facts", []))
    lines.extend(f"- Project decision: {item}" for item in project.get("decisions", []))
    lines.extend(f"- Known issue: {item}" for item in project.get("issues", []))

    summary = session.get("summary", "")
    if summary:
        lines.append(f"- Session summary: {summary}")
    for message in session.get("recent_messages", [])[-6:]:
        lines.append(f"- Recent {message.get('role')}: {message.get('content')}")

    return "\n".join(lines)


def remember_preference(preference: str) -> None:
    profile = load_profile()
    preferences = profile.setdefault("preferences", [])
    if preference not in preferences:
        preferences.append(preference)
        _write_json(PROFILE_PATH, profile)


def remember_project_fact(fact: str) -> None:
    project = load_project_memory()
    facts = project.setdefault("facts", [])
    if fact not in facts:
        facts.append(fact)
        _write_json(PROJECT_MEMORY_PATH, project)


def _session_path(session_id: str) -> Path:
    safe_id = "".join(char for char in session_id if char.isalnum() or char in "-_")
    return SESSIONS_ROOT / f"{safe_id or 'default'}.json"


def _append_summary(summary: str, messages: list[dict[str, str]]) -> str:
    snippets = []
    for message in messages:
        content = message.get("content", "").strip()
        if content:
            snippets.append(f"{message.get('role', 'unknown')}: {content[:160]}")
    combined = " | ".join([summary, *snippets]).strip(" |")
    return combined[-2000:]


def _read_json(path: Path, fallback: dict[str, Any]) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return dict(fallback)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

