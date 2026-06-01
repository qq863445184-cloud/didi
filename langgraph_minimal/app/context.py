from collections.abc import Iterable

from app.memory import load_profile, load_project_memory, load_session

SYSTEM_BUDGET = 1200
PROFILE_BUDGET = 900
PROJECT_BUDGET = 1400
SESSION_BUDGET = 1200
EVIDENCE_BUDGET = 9000


def build_system_context(base_instruction: str, session_id: str = "default") -> str:
    sections = [
        _section("系统指令", [base_instruction], SYSTEM_BUDGET),
        _profile_section(PROFILE_BUDGET),
        _project_section(PROJECT_BUDGET),
        _session_section(session_id, SESSION_BUDGET),
    ]
    return "\n\n".join(section for section in sections if section)


def build_rag_writer_context(
    base_instruction: str,
    question: str,
    evidence: str,
    verification: str = "",
    session_id: str = "default",
) -> tuple[str, str]:
    system_context = build_system_context(base_instruction, session_id=session_id)
    human_sections = [
        _section("当前问题", [question], 1200),
        _section("验证备注", [verification], 800) if verification else "",
        _evidence_section(evidence, EVIDENCE_BUDGET),
    ]
    return system_context, "\n\n".join(section for section in human_sections if section)


def _profile_section(budget: int) -> str:
    profile = load_profile()
    lines = [f"语言：{profile.get('language', 'zh-CN')}"]
    lines.extend(f"用户偏好：{item}" for item in profile.get("preferences", []))
    lines.extend(f"默认配置 {key}：{value}" for key, value in profile.get("defaults", {}).items())
    return _section("用户画像记忆", lines, budget)


def _project_section(budget: int) -> str:
    project = load_project_memory()
    lines = [f"项目：{project.get('project', 'unknown')}"]
    lines.extend(f"项目事实：{item}" for item in project.get("facts", []))
    lines.extend(f"项目决策：{item}" for item in project.get("decisions", []))
    lines.extend(f"已知问题：{item}" for item in project.get("issues", []))
    return _section("项目记忆", lines, budget)


def _session_section(session_id: str, budget: int) -> str:
    session = load_session(session_id)
    lines = []
    if session.get("summary"):
        lines.append(f"会话摘要：{session['summary']}")
    for message in session.get("recent_messages", [])[-6:]:
        lines.append(f"{message.get('role', 'unknown')}: {message.get('content', '')}")
    return _section("会话记忆", lines, budget)


def _section(title: str, lines: Iterable[str], budget: int) -> str:
    unique = _dedupe_lines(lines)
    if not unique:
        return ""
    body = _trim("\n".join(unique), budget)
    return f"[{title}]\n{body}"


def _evidence_section(evidence: str, budget: int) -> str:
    blocks = _split_evidence_blocks(evidence)
    if not blocks:
        return ""

    per_block_budget = max(800, budget // max(1, len(blocks)))
    trimmed_blocks = [_trim(block, per_block_budget) for block in blocks]
    return _section("检索证据", trimmed_blocks, budget)


def _dedupe_lines(lines: Iterable[str]) -> list[str]:
    seen = set()
    unique = []
    for line in lines:
        normalized = " ".join(str(line).strip().split())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique.append(str(line).strip())
    return unique


def _split_evidence_blocks(evidence: str) -> list[str]:
    return [block.strip() for block in evidence.split("\n\n---\n\n") if block.strip()]


def _trim(text: str, budget: int) -> str:
    if len(text) <= budget:
        return text
    return text[:budget].rstrip() + "\n...[truncated]"
