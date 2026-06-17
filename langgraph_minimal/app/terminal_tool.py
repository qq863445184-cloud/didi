from __future__ import annotations

import fnmatch
import os
from pathlib import Path
from typing import Any

from hello_agents.tools.base import Tool, ToolParameter


class TerminalTool(Tool):
    """Read-only filesystem tool for context engineering demos.

    这个工具的目标不是提供任意 shell 执行能力，而是让 Agent 能安全地
    “看看当前项目里有什么”：列目录、读文件、搜文本、取元数据。所有
    路径都会被限制在 root 目录下，避免模型通过 ../ 读到工作区外的文件。
    """

    def __init__(
        self,
        *,
        root: str | Path,
        max_bytes: int = 20_000,
        default_encoding: str = "utf-8",
    ) -> None:
        super().__init__(
            name="terminal",
            description="只读文件系统访问工具，支持 pwd、list、read、search、stat",
        )
        self.root = Path(root).expanduser().resolve()
        self.max_bytes = max_bytes
        self.default_encoding = default_encoding
        self.trace_events: list[dict[str, Any]] = []

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(name="action", type="string", description="操作：pwd、list、read、search、stat", required=True),
            ToolParameter(name="path", type="string", description="相对 root 的路径", required=False, default="."),
            ToolParameter(name="query", type="string", description="搜索关键词，action=search 时使用", required=False),
            ToolParameter(name="pattern", type="string", description="文件匹配模式，例如 *.py 或 *.md", required=False, default="*"),
            ToolParameter(name="limit", type="integer", description="返回条数上限", required=False, default=20),
            ToolParameter(name="max_bytes", type="integer", description="read 的最大读取字节数", required=False),
        ]

    def validate_parameters(self, parameters: dict[str, Any]) -> bool:
        action = parameters.get("action")
        if action not in {"pwd", "list", "read", "search", "stat"}:
            return False
        if action in {"read", "stat"}:
            return bool(str(parameters.get("path", "")).strip())
        if action == "search":
            return bool(str(parameters.get("query", "")).strip())
        return True

    def run(self, parameters: dict[str, Any]) -> str:
        if not self.validate_parameters(parameters):
            self._trace_error("参数不完整或 action 不支持", parameters)
            return "错误：terminal 工具参数不完整或 action 不支持"

        action = str(parameters["action"])
        try:
            if action == "pwd":
                return self._pwd()
            if action == "list":
                return self._list(parameters)
            if action == "read":
                return self._read(parameters)
            if action == "search":
                return self._search(parameters)
            if action == "stat":
                return self._stat(parameters)
        except Exception as exc:
            self._trace_error(str(exc), parameters)
            return f"错误：{exc}"
        self._trace_error(f"不支持的 action={action}", parameters)
        return f"错误：不支持的 action={action}"

    def _pwd(self) -> str:
        self.trace_events.append({"stage": "terminal.pwd", "root": str(self.root)})
        return f"root: {self.root}"

    def _list(self, parameters: dict[str, Any]) -> str:
        path = self._resolve(parameters.get("path", "."))
        if not path.exists():
            raise FileNotFoundError(f"路径不存在: {self._display(path)}")
        if not path.is_dir():
            raise ValueError(f"不是目录: {self._display(path)}")

        limit = int(parameters.get("limit", 20))
        pattern = str(parameters.get("pattern") or "*")
        entries = sorted(path.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower()))
        matched = [entry for entry in entries if fnmatch.fnmatch(entry.name, pattern)][:limit]
        self.trace_events.append(
            {
                "stage": "terminal.list",
                "path": self._display(path),
                "pattern": pattern,
                "count": len(matched),
            }
        )
        if not matched:
            return f"目录为空或无匹配项: {self._display(path)}"

        lines = [f"目录: {self._display(path)}"]
        for entry in matched:
            kind = "dir" if entry.is_dir() else "file"
            size = entry.stat().st_size if entry.is_file() else 0
            lines.append(f"- {entry.name} type={kind} size={size}")
        return "\n".join(lines)

    def _read(self, parameters: dict[str, Any]) -> str:
        path = self._resolve(parameters["path"])
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {self._display(path)}")
        if not path.is_file():
            raise ValueError(f"不是文件: {self._display(path)}")

        raw = path.read_bytes()
        if self._looks_binary(raw):
            raise ValueError(f"疑似二进制文件，拒绝直接读取: {self._display(path)}")

        cap = int(parameters.get("max_bytes") or self.max_bytes)
        clipped = raw[:cap]
        text = clipped.decode(self.default_encoding, errors="replace")
        truncated = len(raw) > cap
        self.trace_events.append(
            {
                "stage": "terminal.read",
                "path": self._display(path),
                "bytes": len(raw),
                "returned_bytes": len(clipped),
                "truncated": truncated,
            }
        )

        header = f"文件: {self._display(path)} bytes={len(raw)}"
        if truncated:
            text += f"\n[truncated: showing first {cap} bytes of {len(raw)} bytes]"
        return f"{header}\n{text}"

    def _search(self, parameters: dict[str, Any]) -> str:
        base = self._resolve(parameters.get("path", "."))
        if not base.exists():
            raise FileNotFoundError(f"路径不存在: {self._display(base)}")
        query = str(parameters["query"]).lower()
        pattern = str(parameters.get("pattern") or "*")
        limit = int(parameters.get("limit", 20))

        files = [base] if base.is_file() else self._iter_files(base, pattern)
        hits: list[str] = []
        scanned = 0
        for file_path in files:
            if len(hits) >= limit:
                break
            scanned += 1
            raw = file_path.read_bytes()
            if self._looks_binary(raw):
                continue
            text = raw.decode(self.default_encoding, errors="replace")
            for line_no, line in enumerate(text.splitlines(), 1):
                if query in line.lower():
                    hits.append(f"- {self._display(file_path)}:{line_no}: {line.strip()}")
                    if len(hits) >= limit:
                        break

        self.trace_events.append(
            {
                "stage": "terminal.search",
                "path": self._display(base),
                "query": parameters["query"],
                "pattern": pattern,
                "scanned": scanned,
                "hits": len(hits),
            }
        )
        if not hits:
            return f"未找到匹配: query={parameters['query']} pattern={pattern}"
        return "\n".join([f"搜索结果: query={parameters['query']}"] + hits)

    def _stat(self, parameters: dict[str, Any]) -> str:
        path = self._resolve(parameters["path"])
        if not path.exists():
            raise FileNotFoundError(f"路径不存在: {self._display(path)}")
        stat = path.stat()
        kind = "dir" if path.is_dir() else "file"
        self.trace_events.append(
            {"stage": "terminal.stat", "path": self._display(path), "type": kind}
        )
        return (
            f"path: {self._display(path)}\n"
            f"type: {kind}\n"
            f"size: {stat.st_size}\n"
            f"modified_at: {stat.st_mtime}"
        )

    def _resolve(self, value: Any) -> Path:
        raw = str(value or ".").strip()
        candidate = (self.root / raw).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise PermissionError("访问被拒绝，路径超出 root") from exc
        return candidate

    def _iter_files(self, base: Path, pattern: str) -> list[Path]:
        return sorted(
            [
                path
                for path in base.rglob("*")
                if path.is_file() and fnmatch.fnmatch(path.name, pattern)
            ],
            key=lambda item: str(item).lower(),
        )

    def _display(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.root)).replace(os.sep, "/") or "."
        except ValueError:
            return str(path)

    def _looks_binary(self, raw: bytes) -> bool:
        sample = raw[:1024]
        return b"\x00" in sample

    def _trace_error(self, error: str, parameters: dict[str, Any]) -> None:
        self.trace_events.append(
            {
                "stage": "terminal.error",
                "error": error,
                "action": parameters.get("action"),
                "path": parameters.get("path"),
            }
        )
