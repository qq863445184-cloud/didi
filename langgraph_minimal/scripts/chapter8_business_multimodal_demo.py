from __future__ import annotations

from pathlib import Path
from typing import Any

from app.my_memory_system import (
    EpisodicMemoryStore,
    MyMemoryManager,
    MyPerceptionTool,
    MyRAGTool,
    PerceptualMemoryStore,
    SemanticMemoryStore,
)


class BusinessKeywordEmbedder:
    """Small deterministic embedder for the business multimodal demo.

    真实环境会换成 sentence-transformers / DashScope / CLIP 等模型；这里用关键词
    维度表达“发票、退款、订单、金额”，让测试可以稳定验证 RAG 编排逻辑。
    """

    KEYWORDS = (
        ("invoice", "发票", "inv-2026-001"),
        ("refund", "退款", "退费"),
        ("order", "订单", "ord-2026-778"),
        ("amount", "金额", "total", "1280"),
    )

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            vectors.append(
                [
                    1.0 if any(keyword in lowered for keyword in group) else 0.0
                    for group in self.KEYWORDS
                ]
            )
        return vectors


class InMemoryVectorStore:
    """Minimal vector store compatible with MyRAGTool tests and demos."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def add_vectors(self, vectors, metadata, ids=None):
        ids = ids or [str(index) for index in range(len(vectors))]
        for vector, meta, row_id in zip(vectors, metadata, ids):
            self.rows.append({"id": row_id, "vector": vector, "metadata": meta})
        return True

    def search_similar(self, query_vector, limit=5, score_threshold=None, where=None):
        hits: list[dict[str, Any]] = []
        for row in self.rows:
            if where and not all(row["metadata"].get(key) == value for key, value in where.items()):
                continue
            score = sum(a * b for a, b in zip(query_vector, row["vector"]))
            if score_threshold is not None and score < score_threshold:
                continue
            hits.append({"id": row["id"], "score": score, "metadata": row["metadata"]})
        hits.sort(key=lambda item: item["score"], reverse=True)
        return hits[:limit]

    def delete_vectors(self, ids=None, where=None):
        """Delete rows either by vector ids or by metadata filter.

        真实 Qdrant 会在删除文档时清理向量索引；本地 demo store 也要遵守
        同一个契约，否则 SQLite 文档清单已删除但检索仍能召回旧 chunk。
        """

        before = len(self.rows)
        id_set = set(ids or [])
        if id_set:
            self.rows = [row for row in self.rows if row["id"] not in id_set]
        elif where:
            self.rows = [
                row
                for row in self.rows
                if not all(row["metadata"].get(key) == value for key, value in where.items())
            ]
        return before - len(self.rows)


class InMemoryGraphStore:
    """Tiny graph-store stand-in so the demo never requires a live Neo4j."""

    def __init__(self) -> None:
        self.entities: dict[str, dict[str, Any]] = {}
        self.relationships: list[dict[str, Any]] = []

    def add_entity(self, entity_id, name, entity_type, properties=None):
        self.entities[entity_id] = {
            "id": entity_id,
            "name": name,
            "type": entity_type,
            **(properties or {}),
        }
        return True

    def add_relationship(self, from_entity_id, to_entity_id, relationship_type, properties=None):
        self.relationships.append(
            {
                "from": from_entity_id,
                "to": to_entity_id,
                "type": relationship_type,
                "properties": properties or {},
            }
        )
        return True

    def search_entities_by_name(self, name_pattern, entity_types=None, limit=5):
        terms = [term.lower() for term in str(name_pattern).split() if term.strip()]
        rows: list[dict[str, Any]] = []
        for entity in self.entities.values():
            if entity_types and entity.get("type") not in entity_types:
                continue
            haystack = " ".join(str(value) for value in entity.values()).lower()
            if not terms or any(term in haystack for term in terms):
                rows.append(entity)
        return rows[:limit]


class BusinessDemoLLM:
    """Fake LLM that makes the answer deterministic while preserving RAG flow."""

    def invoke(self, messages, **kwargs):
        prompt = messages[-1]["content"]
        if "改写成 3 个" in prompt:
            return "INV-2026-001 发票金额\nORD-2026-778 退款请求\ninvoice refund amount"
        if "生成一段可能出现在知识库中的假设性答案" in prompt:
            return "客户针对订单 ORD-2026-778 提出 refund request，并关联发票 INV-2026-001。"
        if "相关上下文" in prompt:
            return self._answer_from_context(prompt)
        return "已从发票图片和客服录音中确认：INV-2026-001 金额 1280 CNY，订单 ORD-2026-778 有 refund request。"

    def _answer_from_context(self, prompt: str) -> str:
        """Generate a deterministic answer from the RAG context.

        The real dashboard can swap this fake LLM for HelloAgentsLLM or another
        provider.  For local tests, this method still follows the same RAG shape:
        answer only from the retrieved context embedded in the prompt.
        """

        question = self._extract_between(prompt, "问题：", "\n\n相关上下文：").strip()
        context = self._extract_between(prompt, "相关上下文：", "\n\n请给出").strip()

        if any(marker in question for marker in ("哪些", "包括", "字段")):
            field_items = self._extract_field_items(context)
            if field_items:
                lead = "统一交互协议可包括：" if "统一交互协议" in question else "相关字段包括："
                return "\n".join([lead, *(f"- {item}" for item in field_items)])

        question_lowered = question.lower()
        if (
            "refund" in question_lowered
            or "invoice" in question_lowered
            or "发票" in question
            or "订单" in question
        ):
            return "根据检索上下文，退款请求关联发票 INV-2026-001，订单号为 ORD-2026-778，金额为 1280 CNY。"

        sentences = self._context_sentences(context)
        if sentences:
            return "\n".join(f"{index}. {sentence}" for index, sentence in enumerate(sentences[:3], 1))
        return "检索上下文中没有足够信息回答该问题。"

    def _extract_between(self, text: str, start: str, end: str) -> str:
        if start not in text:
            return ""
        suffix = text.split(start, 1)[1]
        if end not in suffix:
            return suffix
        return suffix.split(end, 1)[0]

    def _extract_field_items(self, context: str) -> list[str]:
        items: list[str] = []
        for line in context.splitlines():
            normalized = line.strip()
            if not normalized.startswith("- "):
                continue
            normalized = normalized[2:].strip(" ；;。")
            if "：" in normalized:
                key, value = normalized.split("：", 1)
            elif ":" in normalized:
                key, value = normalized.split(":", 1)
            else:
                continue
            key = key.strip(" `")
            value = value.strip(" `；;。")
            if key and value:
                items.append(f"{key}：{value}")
        return items

    def _context_sentences(self, context: str) -> list[str]:
        cleaned = context.replace("片段 ", "\n片段 ")
        parts = []
        for line in cleaned.splitlines():
            line = line.strip()
            if not line or line.startswith("片段"):
                continue
            for sentence in line.replace("；", "。").split("。"):
                sentence = sentence.strip(" #\n\t")
                if sentence and not sentence.startswith("##"):
                    parts.append(sentence)
        return parts


def build_business_multimodal_demo(
    *,
    llm: Any | None = None,
    llm_mode: str = "demo",
) -> tuple[MyPerceptionTool, MyRAGTool, MyMemoryManager]:
    embedder = BusinessKeywordEmbedder()
    rag_tool = MyRAGTool(
        embedder=embedder,
        vector_store=InMemoryVectorStore(),
        llm=llm or BusinessDemoLLM(),
        collection_name="chapter8_business_multimodal",
    )
    rag_tool.llm_mode = llm_mode
    manager = MyMemoryManager(
        user_id="business_demo_user",
        stores={
            "perceptual": PerceptualMemoryStore(),
            "semantic": SemanticMemoryStore(
                embedder=embedder,
                vector_store=InMemoryVectorStore(),
                graph_store=None,
            ),
            "episodic": EpisodicMemoryStore(graph_store=InMemoryGraphStore()),
        },
    )
    perception_tool = MyPerceptionTool(
        manager=manager,
        rag_tool=rag_tool,
        rag_namespace="business_multimodal",
        image_ocr=lambda path: (
            "Invoice No: INV-2026-001. Vendor: Cloud Training Ltd. "
            "Total amount: 1280 CNY. Payment status: pending review."
        ),
        audio_asr=lambda path: (
            "Support call transcript: customer made a refund request for order "
            "ORD-2026-778 because invoice INV-2026-001 amount mismatch."
        ),
    )
    return perception_tool, rag_tool, manager


def run_business_multimodal_demo(work_dir: str | Path) -> dict[str, Any]:
    work_path = Path(work_dir)
    work_path.mkdir(parents=True, exist_ok=True)

    # 这里的文件内容不参与 OCR/ASR；真实 demo 可以替换为真正的图片和录音。
    invoice_image = work_path / "invoice_INV-2026-001.png"
    support_audio = work_path / "support_call_ORD-2026-778.wav"
    invoice_image.write_bytes(b"fake invoice image bytes")
    support_audio.write_bytes(b"fake support audio bytes")

    perception_tool, rag_tool, manager = build_business_multimodal_demo()
    invoice_ingest = perception_tool.run(
        {
            "action": "ingest_file",
            "file_path": str(invoice_image),
            "description": "Supplier invoice screenshot from finance review workflow.",
            "importance": 0.85,
        }
    )
    audio_ingest = perception_tool.run(
        {
            "action": "ingest_file",
            "file_path": str(support_audio),
            "description": "Customer support call recording linked to refund workflow.",
            "importance": 0.9,
        }
    )
    answer = rag_tool.run(
        {
            "action": "ask",
            "question": "Which invoice and order are related to the refund request?",
            "namespace": "business_multimodal",
            "limit": 5,
            "enable_mqe": True,
        }
    )

    return {
        "invoice_ingest": invoice_ingest,
        "audio_ingest": audio_ingest,
        "answer": answer,
        "retrieved_chunks": list(rag_tool.last_retrieved_chunks),
        "trace": [*perception_tool.trace_events, *rag_tool.trace_events],
        "perceptual_count": len(manager.stores["perceptual"].all_records()),
        "semantic_count": len(manager.stores["semantic"].all_records()),
        "episodic_count": len(manager.stores["episodic"].all_records()),
    }


if __name__ == "__main__":
    import json

    result = run_business_multimodal_demo(Path("memory_data") / "business_multimodal_demo")
    print(json.dumps(result, ensure_ascii=False, indent=2))
