from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from math import exp
from typing import Any, Iterable


@dataclass
class ContextPacket:
    """One candidate information unit for context assembly.

    第九章的上下文工程不直接拼大字符串，而是先把系统指令、记忆、
    RAG 证据、对话历史等统一包装成 packet。后续选择、排序、压缩
    都围绕 packet 做，方便 trace 和测试。
    """

    content: str
    source: str = "custom"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    token_count: int = 0
    relevance_score: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.content = str(self.content).strip()
        self.relevance_score = _clamp(self.relevance_score)
        if self.token_count <= 0:
            self.token_count = estimate_tokens(self.content)
        if self.timestamp.tzinfo is None:
            self.timestamp = self.timestamp.replace(tzinfo=timezone.utc)


@dataclass
class ContextConfig:
    """Configuration for the GSSC context pipeline."""

    max_tokens: int = 3000
    reserve_ratio: float = 0.2
    min_relevance: float = 0.1
    enable_compression: bool = True
    recency_weight: float = 0.3
    relevance_weight: float = 0.7
    max_history_messages: int = 5
    packet_compression_floor: int = 80

    def __post_init__(self) -> None:
        if self.max_tokens <= 0:
            raise ValueError("max_tokens 必须大于 0")
        if not 0.0 <= self.reserve_ratio <= 1.0:
            raise ValueError("reserve_ratio 必须在 [0, 1] 范围内")
        if not 0.0 <= self.min_relevance <= 1.0:
            raise ValueError("min_relevance 必须在 [0, 1] 范围内")
        weight_sum = self.recency_weight + self.relevance_weight
        if abs(weight_sum - 1.0) > 1e-6:
            raise ValueError("recency_weight + relevance_weight 必须等于 1.0")


@dataclass
class ContextBuildResult:
    """Structured output from ContextBuilder."""

    context: str
    packets: list[ContextPacket]
    total_tokens: int
    trace: list[dict[str, Any]]


class ContextBuilder:
    """GSSC context builder: Gather -> Select -> Structure -> Compress.

    这个类是第九章的最小可运行内核：
    - Gather：从系统提示、记忆、RAG、历史消息、自定义 packet 汇集候选信息；
    - Select：用相关性 + 新近性评分，在预算内选高价值 packet；
    - Structure：按固定区块输出，便于观察和 A/B 测试；
    - Compress：超预算时做确定性压缩，保证不会无限膨胀。
    """

    def __init__(
        self,
        *,
        config: ContextConfig | None = None,
        memory_tool: Any | None = None,
        rag_tool: Any | None = None,
        now: datetime | None = None,
    ) -> None:
        self.config = config or ContextConfig()
        self.memory_tool = memory_tool
        self.rag_tool = rag_tool
        self.now = now or datetime.now(timezone.utc)
        if self.now.tzinfo is None:
            self.now = self.now.replace(tzinfo=timezone.utc)
        self.trace_events: list[dict[str, Any]] = []

    def build(
        self,
        *,
        user_query: str,
        system_instructions: str = "",
        conversation_history: list[Any] | None = None,
        custom_packets: list[ContextPacket] | None = None,
        output_instructions: str = "",
    ) -> ContextBuildResult:
        query = user_query.strip()
        self.trace_events = []
        packets = self._gather(
            user_query=query,
            system_instructions=system_instructions,
            conversation_history=conversation_history or [],
            custom_packets=custom_packets or [],
        )
        selected = self._select(packets, user_query=query)
        context = self._structure(
            user_query=query,
            selected_packets=selected,
            output_instructions=output_instructions,
        )
        context = self._compress_context(context)
        total_tokens = estimate_tokens(context)
        self.trace_events.append(
            {
                "stage": "context.build",
                "candidate_packets": len(packets),
                "selected_packets": len(selected),
                "total_tokens": total_tokens,
                "max_tokens": self.config.max_tokens,
            }
        )
        return ContextBuildResult(
            context=context,
            packets=selected,
            total_tokens=total_tokens,
            trace=list(self.trace_events),
        )

    def _gather(
        self,
        *,
        user_query: str,
        system_instructions: str,
        conversation_history: list[Any],
        custom_packets: list[ContextPacket],
    ) -> list[ContextPacket]:
        packets: list[ContextPacket] = []
        if system_instructions.strip():
            packets.append(
                ContextPacket(
                    content=system_instructions,
                    source="system",
                    relevance_score=1.0,
                    metadata={"type": "system_instruction", "priority": "high"},
                )
            )

        packets.extend(self._gather_memory(user_query))
        packets.extend(self._gather_rag(user_query))
        packets.extend(self._history_packets(conversation_history))
        packets.extend(custom_packets)

        self.trace_events.append(
            {
                "stage": "context.gather",
                "packets": len(packets),
                "sources": self._count_by_source(packets),
            }
        )
        return packets

    def _gather_memory(self, user_query: str) -> list[ContextPacket]:
        if self.memory_tool is None:
            return []
        try:
            raw = self.memory_tool.run(
                {
                    "action": "search",
                    "query": user_query,
                    "memory_type": "all",
                    "limit": 5,
                }
            )
        except Exception as exc:
            self.trace_events.append(
                {"stage": "context.gather_memory_error", "error": str(exc)}
            )
            return []
        return self._text_to_packets(raw, source="memory", base_score=0.75)

    def _gather_rag(self, user_query: str) -> list[ContextPacket]:
        if self.rag_tool is None:
            return []
        try:
            raw = self.rag_tool.run(
                {
                    "action": "search",
                    "query": user_query,
                    "limit": 5,
                }
            )
        except Exception as exc:
            self.trace_events.append(
                {"stage": "context.gather_rag_error", "error": str(exc)}
            )
            return []
        return self._text_to_packets(raw, source="rag", base_score=0.8)

    def _history_packets(self, history: list[Any]) -> list[ContextPacket]:
        recent = history[-self.config.max_history_messages :]
        packets: list[ContextPacket] = []
        for message in recent:
            role = _message_attr(message, "role", "unknown")
            content = _message_attr(message, "content", "")
            if not str(content).strip():
                continue
            timestamp = _message_attr(message, "timestamp", self.now)
            if not isinstance(timestamp, datetime):
                timestamp = self.now
            packets.append(
                ContextPacket(
                    content=f"{role}: {content}",
                    source="conversation",
                    timestamp=timestamp,
                    relevance_score=0.6,
                    metadata={"type": "conversation_history", "role": role},
                )
            )
        return packets

    def _text_to_packets(
        self,
        raw: Any,
        *,
        source: str,
        base_score: float,
    ) -> list[ContextPacket]:
        text = str(raw or "").strip()
        if not text:
            return []
        blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
        return [
            ContextPacket(
                content=block,
                source=source,
                relevance_score=base_score,
                metadata={"type": source},
            )
            for block in blocks[:8]
        ]

    def _select(
        self,
        packets: list[ContextPacket],
        *,
        user_query: str,
    ) -> list[ContextPacket]:
        reserved_tokens = int(self.config.max_tokens * self.config.reserve_ratio)
        available_tokens = max(0, self.config.max_tokens - reserved_tokens)

        system_packets = [
            packet for packet in packets if packet.metadata.get("priority") == "high"
        ]
        other_packets = [
            packet for packet in packets if packet.metadata.get("priority") != "high"
        ]

        selected: list[ContextPacket] = []
        token_used = 0
        for packet in system_packets:
            selected.append(packet)
            token_used += packet.token_count

        scored: list[tuple[float, ContextPacket]] = []
        for packet in other_packets:
            relevance = packet.relevance_score
            if relevance == 0.5:
                relevance = lexical_relevance(packet.content, user_query)
                packet.relevance_score = relevance
            if relevance < self.config.min_relevance:
                continue
            recency = self._recency_score(packet.timestamp)
            combined = (
                self.config.relevance_weight * relevance
                + self.config.recency_weight * recency
            )
            scored.append((combined, packet))

        scored.sort(key=lambda item: item[0], reverse=True)
        for score, packet in scored:
            if token_used + packet.token_count > available_tokens:
                if not self.config.enable_compression:
                    continue
                compressed = self._compress_packet(packet, available_tokens - token_used)
                if compressed is None:
                    continue
                packet = compressed
            selected.append(packet)
            token_used += packet.token_count
            if token_used >= available_tokens:
                break

        self.trace_events.append(
            {
                "stage": "context.select",
                "available_tokens": available_tokens,
                "selected_packets": len(selected),
                "token_used": token_used,
                "dropped_packets": max(0, len(packets) - len(selected)),
            }
        )
        return selected

    def _structure(
        self,
        *,
        user_query: str,
        selected_packets: list[ContextPacket],
        output_instructions: str,
    ) -> str:
        groups: dict[str, list[ContextPacket]] = {
            "system": [],
            "evidence": [],
            "context": [],
            "output": [],
        }
        for packet in selected_packets:
            section = str(packet.metadata.get("section", "")).strip().lower()
            if section in groups:
                groups[section].append(packet)
                continue
            if packet.source == "system":
                groups["system"].append(packet)
            elif packet.source in {"rag", "memory"}:
                groups["evidence"].append(packet)
            elif packet.source == "conversation":
                groups["context"].append(packet)
            else:
                groups["evidence"].append(packet)

        sections = [
            self._render_section("Role & Policies", groups["system"]),
            f"[Task]\n{user_query}" if user_query else "",
            self._render_section("Evidence", groups["evidence"]),
            self._render_section("Context", groups["context"]),
            self._render_section("Output", groups["output"]),
            f"[Output]\n{output_instructions.strip()}" if output_instructions.strip() else "",
        ]
        context = "\n\n".join(section for section in sections if section)
        self.trace_events.append(
            {
                "stage": "context.structure",
                "sections": [name for name, values in groups.items() if values],
                "tokens": estimate_tokens(context),
            }
        )
        return context

    def _compress_context(self, context: str) -> str:
        tokens = estimate_tokens(context)
        if tokens <= self.config.max_tokens or not self.config.enable_compression:
            return context

        compressed = trim_to_token_budget(context, self.config.max_tokens)
        self.trace_events.append(
            {
                "stage": "context.compress",
                "before_tokens": tokens,
                "after_tokens": estimate_tokens(compressed),
            }
        )
        return compressed

    def _render_section(self, title: str, packets: list[ContextPacket]) -> str:
        if not packets:
            return ""
        lines = []
        for index, packet in enumerate(packets, 1):
            lines.append(
                f"{index}. source={packet.source} score={packet.relevance_score:.2f}\n"
                f"{packet.content}"
            )
        return f"[{title}]\n" + "\n\n".join(lines)

    def _compress_packet(
        self,
        packet: ContextPacket,
        token_budget: int,
    ) -> ContextPacket | None:
        if token_budget < self.config.packet_compression_floor:
            return None
        content = trim_to_token_budget(packet.content, token_budget)
        return ContextPacket(
            content=content,
            source=packet.source,
            timestamp=packet.timestamp,
            relevance_score=packet.relevance_score,
            metadata={**packet.metadata, "compressed": True},
        )

    def _recency_score(self, timestamp: datetime) -> float:
        age_seconds = max(0.0, (self.now - timestamp).total_seconds())
        return exp(-age_seconds / 86400.0)

    def _count_by_source(self, packets: Iterable[ContextPacket]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for packet in packets:
            counts[packet.source] = counts.get(packet.source, 0) + 1
        return counts


def estimate_tokens(text: str) -> int:
    """Small deterministic token estimator for demos and tests.

    不引入 tokenizer 依赖，中文按字符近似，英文按词近似。它不是精确计费
    tokenizer，但足够用于教学版预算守护和单元测试。
    """

    if not text:
        return 0
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
    latin_words = re.findall(r"[A-Za-z0-9_]+", text)
    punctuation = re.findall(r"[^\sA-Za-z0-9_\u4e00-\u9fff]", text)
    return max(1, len(chinese_chars) + len(latin_words) + len(punctuation) // 2)


def lexical_relevance(content: str, query: str) -> float:
    query_terms = set(_tokenize(query))
    if not query_terms:
        return 0.0
    content_terms = set(_tokenize(content))
    if not content_terms:
        return 0.0
    overlap = len(query_terms & content_terms)
    return _clamp(overlap / len(query_terms))


def trim_to_token_budget(text: str, token_budget: int) -> str:
    if estimate_tokens(text) <= token_budget:
        return text
    if token_budget <= 8:
        return "...[compressed]"
    suffix = "\n...[compressed]"
    suffix_tokens = estimate_tokens(suffix)
    target_budget = max(1, token_budget - suffix_tokens)
    chars = max(20, target_budget * 2)
    trimmed = text[:chars].rstrip()
    while chars > 20 and estimate_tokens(trimmed) > target_budget:
        chars = max(20, int(chars * 0.8))
        trimmed = text[:chars].rstrip()
    result = trimmed + suffix
    while estimate_tokens(result) > token_budget and chars > 20:
        chars = max(20, int(chars * 0.9))
        trimmed = text[:chars].rstrip()
        result = trimmed + suffix
    return result


def _tokenize(text: str) -> list[str]:
    lowered = text.lower()
    words = re.findall(r"[a-z0-9_]+", lowered)
    chinese = re.findall(r"[\u4e00-\u9fff]{1,4}", lowered)
    return [*words, *chinese]


def _message_attr(message: Any, name: str, default: Any) -> Any:
    if isinstance(message, dict):
        return message.get(name, default)
    return getattr(message, name, default)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
