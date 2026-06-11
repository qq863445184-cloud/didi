from __future__ import annotations

import re
from typing import Any, Protocol


class EntityExtractor(Protocol):
    """Extract named entities from text for semantic graph indexing."""

    def extract(self, text: str) -> list[dict[str, Any]]:
        ...


class SpacyEntityExtractor:
    """Small spaCy wrapper with a deterministic fallback.

    第八章里 spaCy 用来把非结构化文本变成实体线索，再写入图数据库。
    这里保持同样架构：优先用 spaCy NER；如果本地模型不可用，就只返回空列表，
    不阻塞语义记忆的向量入库。
    """

    def __init__(self, model_names: list[str] | None = None) -> None:
        self.model_names = model_names or ["zh_core_web_sm", "en_core_web_sm"]
        self.nlp = self._load_model()

    def extract(self, text: str) -> list[dict[str, Any]]:
        if self.nlp is None:
            return []

        doc = self.nlp(text)
        entities: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for ent in doc.ents:
            name = ent.text.strip()
            entity_type = ent.label_ or "Entity"
            if not name:
                continue
            key = (name, entity_type)
            if key in seen:
                continue
            entities.append({"name": name, "type": entity_type})
            seen.add(key)
        return entities

    def _load_model(self) -> Any | None:
        try:
            import spacy
        except Exception:
            return None

        for model_name in self.model_names:
            try:
                return spacy.load(model_name)
            except Exception:
                continue
        return None


def normalize_entity_id(name: str, entity_type: str) -> str:
    safe_name = re.sub(r"\s+", "_", name.strip())
    safe_type = re.sub(r"\s+", "_", entity_type.strip() or "Entity")
    return f"entity:{safe_type}:{safe_name}"
