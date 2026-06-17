from __future__ import annotations

import json
import re
from pathlib import Path
from time import time
from typing import Any
from uuid import uuid4

from hello_agents.tools.base import Tool, ToolParameter


class NoteTool(Tool):
    """A small persistent note tool for context engineering demos.

    NoteTool 的定位是“任务笔记”：Agent 可以在推理、阅读、写作过程中
    随手记录阶段性结论，再在后续构建上下文时按关键词检索回来。
    它和 MemoryTool 的区别是更轻量、显式、可人工管理；和 RAGTool 的
    区别是它存的是工作笔记，不负责大文档切块和向量检索。
    """

    def __init__(
        self,
        *,
        path: str | Path = "memory_data/notes.jsonl",
        user_id: str = "default",
    ) -> None:
        super().__init__(
            name="note",
            description="轻量笔记工具，支持保存、搜索、列出、更新和删除任务笔记",
        )
        self.path = Path(path)
        self.user_id = user_id
        self.trace_events: list[dict[str, Any]] = []
        self.notes: list[dict[str, Any]] = self._load()

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(name="action", type="string", description="操作：add、search、list、update、remove、stats", required=True),
            ToolParameter(name="content", type="string", description="笔记正文，action=add/update 时使用", required=False),
            ToolParameter(name="title", type="string", description="笔记标题", required=False),
            ToolParameter(name="query", type="string", description="搜索关键词，action=search 时使用", required=False),
            ToolParameter(name="note_id", type="string", description="笔记 ID，action=update/remove 时使用", required=False),
            ToolParameter(name="tags", type="array", description="标签列表，也支持逗号分隔字符串", required=False),
            ToolParameter(name="importance", type="number", description="笔记重要性，0 到 1", required=False, default=0.5),
            ToolParameter(name="limit", type="integer", description="返回条数", required=False, default=5),
        ]

    def validate_parameters(self, parameters: dict[str, Any]) -> bool:
        action = parameters.get("action")
        if action not in {"add", "search", "list", "update", "remove", "stats"}:
            return False
        if action == "add":
            return bool(str(parameters.get("content", "")).strip())
        if action == "search":
            return bool(str(parameters.get("query", "")).strip())
        if action in {"update", "remove"}:
            return bool(str(parameters.get("note_id", "")).strip())
        return True

    def run(self, parameters: dict[str, Any]) -> str:
        if not self.validate_parameters(parameters):
            return "错误：note 工具参数不完整或 action 不支持"

        action = str(parameters["action"])
        try:
            if action == "add":
                return self._add(parameters)
            if action == "search":
                return self._search(parameters)
            if action == "list":
                return self._list(parameters)
            if action == "update":
                return self._update(parameters)
            if action == "remove":
                return self._remove(parameters)
            if action == "stats":
                return self._stats()
        except Exception as exc:
            return f"错误：{exc}"
        return f"错误：不支持的 action={action}"

    def _add(self, parameters: dict[str, Any]) -> str:
        now = time()
        note = {
            "note_id": str(parameters.get("note_id") or uuid4()),
            "user_id": self.user_id,
            "title": str(parameters.get("title") or "").strip(),
            "content": str(parameters["content"]).strip(),
            "tags": self._normalize_tags(parameters.get("tags")),
            "importance": self._clamp(parameters.get("importance", 0.5)),
            "created_at": now,
            "updated_at": now,
        }
        self.notes.append(note)
        self._save()
        self.trace_events.append(
            {
                "stage": "note.add",
                "note_id": note["note_id"],
                "tags": note["tags"],
            }
        )
        return self._format_note("已保存笔记", note)

    def _search(self, parameters: dict[str, Any]) -> str:
        query = str(parameters["query"]).strip()
        limit = int(parameters.get("limit", 5))
        scored: list[tuple[float, dict[str, Any]]] = []
        for note in self.notes:
            score = self._score_note(note, query)
            if score > 0:
                scored.append((score, note))
        scored.sort(key=lambda item: (item[0], item[1].get("importance", 0.0)), reverse=True)
        hits = scored[:limit]
        self.trace_events.append(
            {
                "stage": "note.search",
                "query": query,
                "hits": len(hits),
            }
        )
        if not hits:
            return f"未找到与 '{query}' 相关的笔记。"

        lines = [f"找到 {len(hits)} 条相关笔记："]
        for index, (score, note) in enumerate(hits, 1):
            lines.append(self._format_note(f"{index}. score={score:.2f}", note))
        return "\n".join(lines)

    def _list(self, parameters: dict[str, Any]) -> str:
        limit = int(parameters.get("limit", 10))
        notes = sorted(self.notes, key=lambda item: item.get("updated_at", 0.0), reverse=True)[:limit]
        self.trace_events.append({"stage": "note.list", "count": len(notes)})
        if not notes:
            return "暂无笔记。"
        lines = [f"笔记列表：{len(notes)} 条"]
        for index, note in enumerate(notes, 1):
            lines.append(self._format_note(f"{index}.", note))
        return "\n".join(lines)

    def _update(self, parameters: dict[str, Any]) -> str:
        note = self._find_note(str(parameters["note_id"]).strip())
        if note is None:
            return f"未找到要更新的笔记：id={parameters['note_id']}"
        if parameters.get("content") is not None:
            note["content"] = str(parameters["content"]).strip()
        if parameters.get("title") is not None:
            note["title"] = str(parameters["title"]).strip()
        if parameters.get("tags") is not None:
            note["tags"] = self._normalize_tags(parameters.get("tags"))
        if parameters.get("importance") is not None:
            note["importance"] = self._clamp(parameters["importance"])
        note["updated_at"] = time()
        self._save()
        self.trace_events.append({"stage": "note.update", "note_id": note["note_id"]})
        return self._format_note("已更新笔记", note)

    def _remove(self, parameters: dict[str, Any]) -> str:
        note_id = str(parameters["note_id"]).strip()
        before = len(self.notes)
        self.notes = [note for note in self.notes if note.get("note_id") != note_id]
        removed = before - len(self.notes)
        self._save()
        self.trace_events.append(
            {"stage": "note.remove", "note_id": note_id, "removed": bool(removed)}
        )
        if not removed:
            return f"未找到要删除的笔记：id={note_id}"
        return f"已删除笔记：id={note_id}"

    def _stats(self) -> str:
        tag_counts: dict[str, int] = {}
        for note in self.notes:
            for tag in note.get("tags", []):
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        self.trace_events.append(
            {"stage": "note.stats", "total": len(self.notes), "tags": tag_counts}
        )
        lines = [f"笔记统计：user_id={self.user_id}", f"total: {len(self.notes)}"]
        for tag, count in sorted(tag_counts.items()):
            lines.append(f"{tag}: {count}")
        return "\n".join(lines)

    def _load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        notes: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if payload.get("user_id") == self.user_id:
                notes.append(payload)
        return notes

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        rows = [json.dumps(note, ensure_ascii=False) for note in self.notes]
        self.path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")

    def _find_note(self, note_id: str) -> dict[str, Any] | None:
        for note in self.notes:
            if note.get("note_id") == note_id:
                return note
        return None

    def _score_note(self, note: dict[str, Any], query: str) -> float:
        query_terms = self._terms(query)
        if not query_terms:
            return 0.0
        haystack = " ".join(
            [
                str(note.get("title", "")),
                str(note.get("content", "")),
                " ".join(str(tag) for tag in note.get("tags", [])),
            ]
        )
        overlap = len(query_terms & self._terms(haystack))
        if overlap <= 0:
            return 0.0
        return overlap / len(query_terms) + float(note.get("importance", 0.5)) * 0.1

    def _terms(self, text: str) -> set[str]:
        normalized = text.lower()
        terms = set(re.findall(r"[a-z0-9][a-z0-9_-]*|[\u4e00-\u9fff]+", normalized))
        for term in list(terms):
            if re.fullmatch(r"[\u4e00-\u9fff]+", term):
                terms.update(term[index : index + 2] for index in range(max(0, len(term) - 1)))
        return terms

    def _normalize_tags(self, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            raw = re.split(r"[,，\s]+", value)
        elif isinstance(value, list):
            raw = [str(item) for item in value]
        else:
            raw = [str(value)]
        tags: list[str] = []
        seen: set[str] = set()
        for item in raw:
            tag = item.strip()
            if tag and tag not in seen:
                tags.append(tag)
                seen.add(tag)
        return tags

    def _format_note(self, prefix: str, note: dict[str, Any]) -> str:
        title = note.get("title") or "<untitled>"
        tags = ", ".join(note.get("tags", [])) or "-"
        return (
            f"{prefix}\n"
            f"- id: {note.get('note_id')}\n"
            f"- title: {title}\n"
            f"- tags: {tags}\n"
            f"- importance: {float(note.get('importance', 0.0)):.2f}\n"
            f"- content: {note.get('content', '')}"
        )

    def _clamp(self, value: Any) -> float:
        return max(0.0, min(1.0, float(value)))
