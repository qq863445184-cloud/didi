from __future__ import annotations

import os
from pathlib import Path
from time import time
from typing import Any
from uuid import uuid4

from hello_agents.tools.base import Tool, ToolParameter


class MyRAGTool(Tool):
    """A small hello-agents compatible RAG tool.

    和 hello-agents 原生 RAGTool 相比，这个版本刻意保持教学透明：
    add_text 不写临时文件，直接切块；search 明确按 namespace 过滤；
    ask 明确展示“检索上下文 -> LLM 生成”的流程。
    """

    def __init__(
        self,
        *,
        embedder: Any | None = None,
        vector_store: Any | None = None,
        llm: Any | None = None,
        collection_name: str = "my_rag_knowledge_base",
        vector_size: int = 384,
    ) -> None:
        super().__init__(
            name="rag",
            description="自定义 RAG 工具，支持文本入库、语义检索和基于知识库问答",
        )
        self.collection_name = collection_name
        self.embedder = embedder or self._build_default_embedder()
        self.vector_store = vector_store or self._build_default_vector_store(
            collection_name=collection_name,
            vector_size=vector_size,
        )
        self.llm = llm or self._build_default_llm()
        self.trace_events: list[dict[str, Any]] = []
        self.added_chunks = 0

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(name="action", type="string", description="操作：add_text、add_document、search、ask、stats", required=True),
            ToolParameter(name="text", type="string", description="要加入知识库的文本", required=False),
            ToolParameter(name="file_path", type="string", description="要加入知识库的文档路径", required=False),
            ToolParameter(name="document_id", type="string", description="文档 ID", required=False),
            ToolParameter(name="query", type="string", description="检索 query", required=False),
            ToolParameter(name="question", type="string", description="问答问题", required=False),
            ToolParameter(name="namespace", type="string", description="知识库命名空间", required=False, default="default"),
            ToolParameter(name="limit", type="integer", description="返回条数", required=False, default=5),
            ToolParameter(name="chunk_size", type="integer", description="切块字符数", required=False, default=800),
            ToolParameter(name="chunk_overlap", type="integer", description="相邻块重叠字符数", required=False, default=100),
        ]

    def validate_parameters(self, parameters: dict[str, Any]) -> bool:
        action = parameters.get("action")
        if action == "add_text":
            return bool(str(parameters.get("text", "")).strip())
        if action == "add_document":
            return bool(str(parameters.get("file_path", "")).strip())
        if action == "search":
            return bool(str(parameters.get("query") or parameters.get("question") or "").strip())
        if action == "ask":
            return bool(str(parameters.get("question") or parameters.get("query") or "").strip())
        if action == "stats":
            return True
        return False

    def run(self, parameters: dict[str, Any]) -> str:
        if not self.validate_parameters(parameters):
            return "错误：rag 工具参数不完整或 action 不支持"

        action = parameters["action"]
        try:
            if action == "add_text":
                return self._add_text(parameters)
            if action == "add_document":
                return self._add_document(parameters)
            if action == "search":
                return self._search(parameters)
            if action == "ask":
                return self._ask(parameters)
            if action == "stats":
                return self._stats()
            return f"错误：不支持的 action={action}"
        except Exception as exc:
            return f"错误：{exc}"

    def _add_text(self, parameters: dict[str, Any]) -> str:
        text = str(parameters["text"]).strip()
        namespace = str(parameters.get("namespace", "default"))
        document_id = str(parameters.get("document_id") or f"doc_{uuid4()}")
        chunk_size = int(parameters.get("chunk_size", 800))
        chunk_overlap = int(parameters.get("chunk_overlap", 100))

        chunks = self._chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        vectors = self._encode_many(chunks)
        ids = [str(uuid4()) for _ in chunks]
        metadata = [
            {
                "content": chunk,
                "document_id": document_id,
                "namespace": namespace,
                "chunk_index": index,
                "memory_type": "rag_chunk",
                "data_source": "my_rag_tool",
                "created_at": time(),
            }
            for index, chunk in enumerate(chunks)
        ]
        ok = self.vector_store.add_vectors(vectors=vectors, metadata=metadata, ids=ids)
        if ok is False:
            raise RuntimeError("RAG 向量写入失败")

        self.added_chunks += len(chunks)
        self.trace_events.append(
            {
                "stage": "rag.add_text",
                "document_id": document_id,
                "namespace": namespace,
                "chunks": len(chunks),
            }
        )
        return f"文本已添加到知识库: {document_id}\n分块数量: {len(chunks)}\n命名空间: {namespace}"

    def _add_document(self, parameters: dict[str, Any]) -> str:
        file_path = Path(str(parameters["file_path"])).expanduser()
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        if not file_path.is_file():
            raise ValueError(f"不是文件: {file_path}")

        text, parser = self._load_document_text(file_path)
        if not text.strip():
            raise ValueError(f"文档没有提取出可索引文本: {file_path}")

        # 文件入口最终复用 add_text 的切块与向量写入逻辑，避免两套入库路径分叉。
        delegated = {
            **parameters,
            "action": "add_text",
            "text": text,
            "document_id": str(parameters.get("document_id") or file_path.name),
        }
        result = self._add_text(delegated)
        self.trace_events[-1] = {
            **self.trace_events[-1],
            "stage": "rag.add_document",
            "file_path": str(file_path),
            "parser": parser,
        }
        return (
            f"文档已添加到知识库: {file_path.name}\n"
            f"解析器: {parser}\n"
            f"{result}"
        )

    def _search(self, parameters: dict[str, Any]) -> str:
        query = str(parameters.get("query") or parameters.get("question")).strip()
        namespace = str(parameters.get("namespace", "default"))
        limit = int(parameters.get("limit", 5))
        results = self._retrieve(query=query, namespace=namespace, limit=limit)

        self.trace_events.append(
            {
                "stage": "rag.search",
                "query": query,
                "namespace": namespace,
                "hits": len(results),
            }
        )
        if not results:
            return f"未找到与 '{query}' 相关的内容。"

        lines = ["搜索结果："]
        for index, item in enumerate(results, 1):
            meta = item["metadata"]
            content = meta.get("content", "")
            lines.append(
                f"{index}. document={meta.get('document_id')} "
                f"chunk={meta.get('chunk_index')} score={item['score']:.3f}\n"
                f"   {content}"
            )
        return "\n".join(lines)

    def _ask(self, parameters: dict[str, Any]) -> str:
        question = str(parameters.get("question") or parameters.get("query")).strip()
        namespace = str(parameters.get("namespace", "default"))
        limit = int(parameters.get("limit", 5))
        results = self._retrieve(query=question, namespace=namespace, limit=limit)

        if not results:
            return f"未找到与 '{question}' 相关的知识库内容。"

        context = "\n\n".join(
            f"片段 {index}: {item['metadata'].get('content', '')}"
            for index, item in enumerate(results, 1)
        )
        prompt = [
            {
                "role": "system",
                "content": "你是一个 RAG 问答助手。必须基于提供的相关上下文回答，不要编造。",
            },
            {
                "role": "user",
                "content": f"问题：{question}\n\n相关上下文：\n{context}\n\n请给出简洁准确的回答。",
            },
        ]
        answer = self.llm.invoke(prompt)
        self.trace_events.append(
            {
                "stage": "rag.ask",
                "question": question,
                "namespace": namespace,
                "hits": len(results),
            }
        )

        lines = ["智能问答结果：", str(answer).strip(), "", "参考来源："]
        for index, item in enumerate(results, 1):
            meta = item["metadata"]
            lines.append(
                f"{index}. {meta.get('document_id')}#chunk-{meta.get('chunk_index')} "
                f"score={item['score']:.3f}"
            )
        return "\n".join(lines)

    def _stats(self) -> str:
        return f"RAG collection={self.collection_name}, 当前进程已添加 chunk 数={self.added_chunks}"

    def _retrieve(self, *, query: str, namespace: str, limit: int) -> list[dict[str, Any]]:
        query_vector = self._encode_one(query)
        hits = self.vector_store.search_similar(
            query_vector=query_vector,
            limit=limit,
            where={
                "namespace": namespace,
                "memory_type": "rag_chunk",
                "data_source": "my_rag_tool",
            },
        )
        results: list[dict[str, Any]] = []
        seen_contents: set[str] = set()
        for hit in hits:
            meta = hit.get("metadata", {}) or {}
            content = str(meta.get("content", ""))
            if not content or content in seen_contents:
                continue
            seen_contents.add(content)
            results.append(
                {
                    "id": hit.get("id"),
                    "score": float(hit.get("score", 0.0)),
                    "metadata": meta,
                }
            )
        return results[:limit]

    def _chunk_text(self, text: str, *, chunk_size: int, chunk_overlap: int) -> list[str]:
        chunk_size = max(1, chunk_size)
        chunk_overlap = max(0, min(chunk_overlap, chunk_size - 1))
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = min(len(text), start + chunk_size)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(text):
                break
            start = end - chunk_overlap
        return chunks

    def _load_document_text(self, file_path: Path) -> tuple[str, str]:
        suffix = file_path.suffix.lower()
        if suffix in {".txt", ".md", ".markdown", ".json", ".csv", ".log", ".py"}:
            return file_path.read_text(encoding="utf-8"), "plain_text"

        # 第八章文档管线通常会用 MarkItDown/Unstructured 这类解析器。
        # 这里把 MarkItDown 做成可选能力：安装了就解析 PDF/Office，没安装就给出清晰提示。
        try:
            from markitdown import MarkItDown
        except Exception as exc:
            raise RuntimeError(
                f"当前文件类型 {suffix or '<无后缀>'} 需要安装 markitdown 后才能解析"
            ) from exc

        converted = MarkItDown().convert(str(file_path))
        text = getattr(converted, "text_content", None) or str(converted)
        return text, "markitdown"

    def _encode_many(self, texts: list[str]) -> list[list[float]]:
        encoded = self.embedder.encode(texts)
        if hasattr(encoded, "tolist"):
            encoded = encoded.tolist()
        vectors: list[list[float]] = []
        for item in encoded:
            if hasattr(item, "tolist"):
                item = item.tolist()
            vectors.append([float(value) for value in item])
        return vectors

    def _encode_one(self, text: str) -> list[float]:
        return self._encode_many([text])[0]

    def _build_default_embedder(self) -> Any:
        from hello_agents.memory.embedding import get_text_embedder

        return get_text_embedder()

    def _build_default_vector_store(self, *, collection_name: str, vector_size: int) -> Any:
        from hello_agents.memory.storage.qdrant_store import QdrantVectorStore

        return QdrantVectorStore(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY") or None,
            collection_name=collection_name,
            vector_size=vector_size,
            distance="cosine",
        )

    def _build_default_llm(self) -> Any:
        from hello_agents import HelloAgentsLLM

        return HelloAgentsLLM()
