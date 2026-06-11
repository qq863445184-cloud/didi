from __future__ import annotations

from typing import Any

from hello_agents.tools.base import Tool, ToolParameter

from .manager import MyMemoryManager
from .models import MemoryRecord, MemorySearchResult


class MyMemoryTool(Tool):
    """hello-agents compatible memory tool backed by our own manager.

    它仍然是 hello-agents 的 Tool，所以可以被 ToolRegistry 或自定义 Agent
    当作普通工具使用；但真正的记忆逻辑完全在 app/my_memory_system 内部。
    """

    def __init__(
        self,
        *,
        user_id: str = "default",
        manager: MyMemoryManager | None = None,
    ) -> None:
        super().__init__(
            name="memory",
            description="自定义记忆工具，支持添加、检索和摘要用户记忆",
        )
        self.manager = manager or MyMemoryManager(user_id=user_id)
        self.trace_events: list[dict[str, Any]] = []

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="action",
                type="string",
                description="操作类型：add、search、summary、forget、consolidate",
                required=True,
            ),
            ToolParameter(
                name="content",
                type="string",
                description="需要保存的记忆内容，action=add 时使用",
                required=False,
            ),
            ToolParameter(
                name="query",
                type="string",
                description="检索关键词或问题，action=search 时使用",
                required=False,
            ),
            ToolParameter(
                name="memory_type",
                type="string",
                description="记忆类型，当前支持 working，可扩展 semantic、episodic、perceptual",
                required=False,
                default="working",
            ),
            ToolParameter(
                name="importance",
                type="number",
                description="记忆重要性，0 到 1",
                required=False,
                default=0.5,
            ),
            ToolParameter(
                name="limit",
                type="integer",
                description="返回结果数量",
                required=False,
                default=5,
            ),
            ToolParameter(
                name="strategy",
                type="string",
                description="遗忘策略：importance、age、capacity，action=forget 时使用",
                required=False,
                default="importance",
            ),
            ToolParameter(
                name="importance_threshold",
                type="number",
                description="重要性阈值；forget 删除低于它的记忆，consolidate 整合不低于它的记忆",
                required=False,
            ),
            ToolParameter(
                name="max_age_seconds",
                type="number",
                description="遗忘的最大存活秒数，strategy=age 时使用",
                required=False,
            ),
            ToolParameter(
                name="capacity",
                type="integer",
                description="保留的记忆条数上限，strategy=capacity 时使用",
                required=False,
            ),
            ToolParameter(
                name="target_type",
                type="string",
                description="整合目标记忆类型，action=consolidate 时使用，默认 semantic",
                required=False,
                default="semantic",
            ),
        ]

    def validate_parameters(self, parameters: dict[str, Any]) -> bool:
        action = parameters.get("action")
        if action not in {"add", "search", "summary", "forget", "consolidate"}:
            return False
        if action == "add":
            return bool(str(parameters.get("content", "")).strip())
        if action == "search":
            return bool(str(parameters.get("query", "")).strip())
        return True

    def run(self, parameters: dict[str, Any]) -> str:
        action = parameters.get("action")
        memory_type = parameters.get("memory_type", "working")

        try:
            if not self.validate_parameters(parameters):
                return "错误：memory 工具参数不完整或 action 不支持"

            if action == "add":
                result = self._add(parameters, memory_type)
            elif action == "search":
                result = self._search(parameters, memory_type)
            elif action == "summary":
                result = self._summary(parameters, memory_type)
            elif action == "forget":
                result = self._forget(parameters, memory_type)
            elif action == "consolidate":
                result = self._consolidate(parameters, memory_type)
            else:
                result = f"错误：不支持的 action={action}"

            self.trace_events = list(self.manager.trace_events)
            return result
        except Exception as exc:
            self.trace_events = list(self.manager.trace_events)
            return f"错误：{exc}"

    def _add(self, parameters: dict[str, Any], memory_type: str) -> str:
        record = self.manager.add(
            content=str(parameters["content"]).strip(),
            memory_type=memory_type,
            importance=float(parameters.get("importance", 0.5)),
            metadata=self._extract_metadata(parameters),
        )
        return self._format_add_result(record)

    def _search(self, parameters: dict[str, Any], memory_type: str) -> str:
        results = self.manager.search(
            query=str(parameters["query"]).strip(),
            memory_type=memory_type,
            limit=int(parameters.get("limit", 5)),
        )
        return self._format_search_result(results)

    def _summary(self, parameters: dict[str, Any], memory_type: str) -> str:
        records = self.manager.summary(
            memory_type=memory_type,
            limit=int(parameters.get("limit", 5)),
        )
        return self._format_summary_result(records)

    def _forget(self, parameters: dict[str, Any], memory_type: str) -> str:
        strategy = str(parameters.get("strategy", "importance"))
        kwargs: dict[str, Any] = {"memory_type": memory_type, "strategy": strategy}
        if parameters.get("importance_threshold") is not None:
            kwargs["importance_threshold"] = float(parameters["importance_threshold"])
        if parameters.get("max_age_seconds") is not None:
            kwargs["max_age_seconds"] = float(parameters["max_age_seconds"])
        if parameters.get("capacity") is not None:
            kwargs["capacity"] = int(parameters["capacity"])

        forgotten = self.manager.forget(**kwargs)
        return self._format_forget_result(forgotten, memory_type, strategy)

    def _consolidate(self, parameters: dict[str, Any], memory_type: str) -> str:
        kwargs: dict[str, Any] = {
            "source_type": memory_type,
            "target_type": str(parameters.get("target_type", "semantic")),
        }
        if parameters.get("importance_threshold") is not None:
            kwargs["importance_threshold"] = float(parameters["importance_threshold"])

        consolidated = self.manager.consolidate(**kwargs)
        return self._format_consolidate_result(consolidated, kwargs["target_type"])

    def _extract_metadata(self, parameters: dict[str, Any]) -> dict[str, Any]:
        reserved = {
            "action",
            "content",
            "query",
            "memory_type",
            "importance",
            "limit",
            "strategy",
            "importance_threshold",
            "max_age_seconds",
            "capacity",
            "target_type",
        }
        return {key: value for key, value in parameters.items() if key not in reserved}

    def _format_add_result(self, record: MemoryRecord) -> str:
        return (
            "已保存记忆：\n"
            f"- id: {record.memory_id}\n"
            f"- type: {record.memory_type}\n"
            f"- importance: {record.importance:.2f}\n"
            f"- content: {record.content}"
        )

    def _format_search_result(self, results: list[MemorySearchResult]) -> str:
        if not results:
            return "未找到相关记忆。"

        lines = [f"找到 {len(results)} 条相关记忆："]
        for index, item in enumerate(results, 1):
            metadata_parts = []
            for key in ("modality", "file_path", "extracted_text"):
                value = item.record.metadata.get(key)
                if value:
                    metadata_parts.append(f"{key}={value}")
            metadata_text = f" metadata: {'; '.join(metadata_parts)}" if metadata_parts else ""
            lines.append(
                f"{index}. [{item.source}] score={item.score:.2f} "
                f"type={item.record.memory_type} content={item.record.content}{metadata_text}"
            )
        return "\n".join(lines)

    def _format_summary_result(self, records: list[MemoryRecord]) -> str:
        lines = [f"记忆总数: {len(records)}"]
        for index, record in enumerate(records, 1):
            lines.append(
                f"{index}. type={record.memory_type} "
                f"importance={record.importance:.2f} content={record.content}"
            )
        return "\n".join(lines)

    def _format_forget_result(
        self, forgotten: list[MemoryRecord], memory_type: str, strategy: str
    ) -> str:
        lines = [f"已遗忘 {len(forgotten)} 条记忆（type={memory_type}, strategy={strategy}）："]
        for index, record in enumerate(forgotten, 1):
            lines.append(
                f"{index}. importance={record.importance:.2f} content={record.content}"
            )
        return "\n".join(lines)

    def _format_consolidate_result(
        self, consolidated: list[MemoryRecord], target_type: str
    ) -> str:
        lines = [f"已整合 {len(consolidated)} 条记忆到 {target_type}："]
        for index, record in enumerate(consolidated, 1):
            lines.append(
                f"{index}. importance={record.importance:.2f} content={record.content}"
            )
        return "\n".join(lines)
