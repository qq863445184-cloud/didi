from __future__ import annotations

import os
import re
from pathlib import Path
from time import time
from typing import Any
from uuid import uuid4

from hello_agents.tools.base import Tool, ToolParameter

from .document_parser import DocumentParserPipeline


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
        document_store: Any | None = None,
        parser_pipeline: DocumentParserPipeline | None = None,
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
        self.document_store = document_store
        self.parser_pipeline = parser_pipeline or DocumentParserPipeline()
        self.trace_events: list[dict[str, Any]] = []
        self.last_retrieved_chunks: list[dict[str, Any]] = []
        self.added_chunks = 0

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(name="action", type="string", description="操作：add_text、add_document、delete_document、search、ask、stats", required=True),
            ToolParameter(name="text", type="string", description="要加入知识库的文本", required=False),
            ToolParameter(name="file_path", type="string", description="要加入知识库的文档路径", required=False),
            ToolParameter(name="document_id", type="string", description="文档 ID", required=False),
            ToolParameter(name="query", type="string", description="检索 query", required=False),
            ToolParameter(name="question", type="string", description="问答问题", required=False),
            ToolParameter(name="namespace", type="string", description="知识库命名空间", required=False, default="default"),
            ToolParameter(name="limit", type="integer", description="返回条数", required=False, default=5),
            ToolParameter(name="chunk_size", type="integer", description="切块字符数", required=False, default=800),
            ToolParameter(name="chunk_overlap", type="integer", description="相邻块重叠字符数", required=False, default=100),
            ToolParameter(name="enable_mqe", type="boolean", description="是否启用多查询扩展检索", required=False, default=False),
            ToolParameter(name="enable_hyde", type="boolean", description="是否启用 HyDE 假设答案检索", required=False, default=False),
            ToolParameter(name="enable_keyword_rerank", type="boolean", description="是否启用关键词二阶段重排", required=False, default=False),
            ToolParameter(name="keyword_weight", type="number", description="关键词重排分数权重", required=False, default=0.2),
            ToolParameter(name="score_threshold", type="number", description="最低相似度阈值", required=False),
            ToolParameter(name="candidate_pool_size", type="integer", description="每个候选查询的召回池大小", required=False),
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
        if action == "delete_document":
            return bool(str(parameters.get("document_id", "")).strip())
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
            if action == "delete_document":
                return self._delete_document(parameters)
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

        chunks = self._build_chunks(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunk_texts = [chunk["content"] for chunk in chunks]
        vectors = self._encode_many(chunk_texts)
        ids = [str(uuid4()) for _ in chunks]
        metadata = [
            {
                "content": chunk["content"],
                "document_id": document_id,
                "namespace": namespace,
                "chunk_index": index,
                "section_title": chunk.get("section_title"),
                "start_char": chunk.get("start_char"),
                "end_char": chunk.get("end_char"),
                "memory_type": "rag_chunk",
                "data_source": "my_rag_tool",
                "created_at": time(),
            }
            for index, chunk in enumerate(chunks)
        ]
        ok = self.vector_store.add_vectors(vectors=vectors, metadata=metadata, ids=ids)
        if ok is False:
            raise RuntimeError("RAG 向量写入失败")

        self._sync_document_store(
            document_id=document_id,
            namespace=namespace,
            chunks=chunks,
            ids=ids,
            parser=str(parameters.get("parser") or "text"),
            source_path=parameters.get("source_path"),
        )
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

        parsed = self.parser_pipeline.parse(file_path)
        text = parsed.text
        parser = parsed.parser
        if not text.strip():
            raise ValueError(f"文档没有提取出可索引文本: {file_path}")

        # 文件入口最终复用 add_text 的切块与向量写入逻辑，避免两套入库路径分叉。
        delegated = {
            **parameters,
            "action": "add_text",
            "text": text,
            "document_id": str(parameters.get("document_id") or file_path.name),
            "source_path": str(file_path),
            "parser": parser,
            "modality": parsed.modality,
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

    def _sync_document_store(
        self,
        *,
        document_id: str,
        namespace: str,
        chunks: list[dict[str, Any]],
        ids: list[str],
        parser: str,
        source_path: Any = None,
    ) -> None:
        if self.document_store is None:
            return

        self.document_store.upsert_document(
            document_id=document_id,
            namespace=namespace,
            source_path=str(source_path) if source_path else None,
            parser=parser,
            metadata={
                "collection_name": self.collection_name,
                "chunk_count": len(chunks),
            },
        )
        self.document_store.replace_chunks(
            document_id=document_id,
            namespace=namespace,
            chunks=[
                {
                    "chunk_id": ids[index],
                    "chunk_index": index,
                    "content": chunk.get("content", ""),
                    "section_title": chunk.get("section_title"),
                    "start_char": chunk.get("start_char"),
                    "end_char": chunk.get("end_char"),
                }
                for index, chunk in enumerate(chunks)
            ],
        )

    def _delete_document(self, parameters: dict[str, Any]) -> str:
        document_id = str(parameters["document_id"]).strip()
        namespace = str(parameters.get("namespace", "default"))
        chunks = (
            self.document_store.list_chunks(document_id, namespace=namespace)
            if self.document_store is not None
            else []
        )

        vector_deleted = self._delete_document_vectors(
            document_id=document_id,
            namespace=namespace,
            chunk_ids=[str(chunk["chunk_id"]) for chunk in chunks],
        )
        metadata_deleted = (
            self.document_store.delete_document(document_id, namespace=namespace)
            if self.document_store is not None
            else False
        )

        self.trace_events.append(
            {
                "stage": "rag.delete_document",
                "document_id": document_id,
                "namespace": namespace,
                "metadata_deleted": bool(metadata_deleted),
                "vector_deleted": vector_deleted,
            }
        )
        if metadata_deleted or vector_deleted:
            return (
                f"文档已删除: {document_id}\n"
                f"命名空间: {namespace}\n"
                f"删除向量数: {vector_deleted}"
            )
        return f"未找到要删除的文档: {document_id}"

    def _delete_document_vectors(
        self,
        *,
        document_id: str,
        namespace: str,
        chunk_ids: list[str],
    ) -> int:
        """Delete vectors for one RAG document across common vector-store APIs."""

        if chunk_ids:
            delete_vectors = getattr(self.vector_store, "delete_vectors", None)
            if callable(delete_vectors):
                result = None
                try:
                    result = delete_vectors(chunk_ids)
                except TypeError:
                    try:
                        result = delete_vectors(ids=chunk_ids)
                    except TypeError:
                        result = None
                if isinstance(result, bool):
                    return len(chunk_ids) if result else 0
                if isinstance(result, int):
                    return result

        filter_delete = getattr(self.vector_store, "delete_vectors", None)
        if callable(filter_delete):
            try:
                result = filter_delete(
                    where={
                        "document_id": document_id,
                        "namespace": namespace,
                        "memory_type": "rag_chunk",
                    }
                )
            except TypeError:
                result = None
            if isinstance(result, bool):
                return len(chunk_ids) if result else 0
            if isinstance(result, int):
                return result

        return 0

    def _search(self, parameters: dict[str, Any]) -> str:
        query = str(parameters.get("query") or parameters.get("question")).strip()
        namespace = str(parameters.get("namespace", "default"))
        limit = int(parameters.get("limit", 5))
        results = self._retrieve_with_options(
            query=query,
            namespace=namespace,
            limit=limit,
            parameters=parameters,
        )
        self.last_retrieved_chunks = self._format_retrieved_chunks(results)
        candidate_queries = self._last_candidate_query_count(query)

        self.trace_events.append(
            {
                "stage": "rag.search",
                "query": query,
                "namespace": namespace,
                "hits": len(results),
                "candidate_queries": candidate_queries,
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
        results = self._retrieve_with_options(
            query=question,
            namespace=namespace,
            limit=limit,
            parameters=parameters,
        )
        answer_results = self._select_answer_results(results)
        self.last_retrieved_chunks = self._format_retrieved_chunks(answer_results)

        if not answer_results:
            return f"未找到与 '{question}' 相关的知识库内容。"

        context = "\n\n".join(
            f"片段 {index}: {item['metadata'].get('content', '')}"
            for index, item in enumerate(answer_results, 1)
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
                "hits": len(answer_results),
            }
        )

        lines = ["智能问答结果：", str(answer).strip(), "", "参考来源："]
        for index, item in enumerate(answer_results, 1):
            meta = item["metadata"]
            lines.append(
                f"{index}. {meta.get('document_id')}#chunk-{meta.get('chunk_index')} "
                f"score={item['score']:.3f}"
            )
        return "\n".join(lines)

    def _select_answer_results(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return the chunks that are actually used to augment an answer."""

        relevant = [
            item
            for item in results
            if float(item.get("score", 0.0)) > 0.0
            or float(item.get("vector_score", 0.0)) > 0.0
            or float(item.get("keyword_score", 0.0)) > 0.0
        ]
        return relevant or results

    def _format_retrieved_chunks(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Expose retrieved chunk text for demos, debugging, and UI evidence panels."""

        chunks: list[dict[str, Any]] = []
        for item in results:
            meta = item.get("metadata", {}) or {}
            chunks.append(
                {
                    "id": item.get("id"),
                    "score": float(item.get("score", 0.0)),
                    "vector_score": float(item.get("vector_score", item.get("score", 0.0))),
                    "keyword_score": float(item.get("keyword_score", 0.0)),
                    "document_id": meta.get("document_id"),
                    "chunk_index": meta.get("chunk_index"),
                    "section_title": meta.get("section_title"),
                    "content": meta.get("content", ""),
                }
            )
        return chunks

    def _stats(self) -> str:
        return f"RAG collection={self.collection_name}, 当前进程已添加 chunk 数={self.added_chunks}"

    def _retrieve(self, *, query: str, namespace: str, limit: int) -> list[dict[str, Any]]:
        return self._retrieve_once(
            query=query,
            namespace=namespace,
            limit=limit,
            score_threshold=None,
        )

    def _retrieve_with_options(
        self,
        *,
        query: str,
        namespace: str,
        limit: int,
        parameters: dict[str, Any],
    ) -> list[dict[str, Any]]:
        candidate_queries = self._build_candidate_queries(query, parameters)
        score_threshold = (
            float(parameters["score_threshold"])
            if parameters.get("score_threshold") is not None
            else None
        )
        if (
            not self._as_bool(parameters.get("enable_mqe"))
            and not self._as_bool(parameters.get("enable_hyde"))
        ):
            recall_limit = (
                int(parameters.get("candidate_pool_size"))
                if parameters.get("candidate_pool_size")
                else max(limit * 4, limit)
            )
            results = self._retrieve_once(
                query=query,
                namespace=namespace,
                limit=recall_limit
                if self._as_bool(parameters.get("enable_keyword_rerank"))
                else limit,
                score_threshold=score_threshold,
            )
            return self._maybe_keyword_rerank(
                query=query,
                results=results,
                limit=limit,
                parameters=parameters,
            )

        pool_size = int(parameters.get("candidate_pool_size") or max(limit * 2, limit))

        candidates: list[dict[str, Any]] = []
        for candidate_query in candidate_queries:
            candidates.extend(
                self._retrieve_once(
                    query=candidate_query,
                    namespace=namespace,
                    limit=pool_size,
                    score_threshold=score_threshold,
                )
            )

        merged = self._merge_candidates(candidates)
        self.trace_events.append(
            {
                "stage": "rag.merge_candidates",
                "query": query,
                "candidate_queries": len(candidate_queries),
                "raw_candidates": len(candidates),
                "merged_candidates": len(merged),
                "score_threshold": score_threshold,
            }
        )
        return self._maybe_keyword_rerank(
            query=query,
            results=merged,
            limit=limit,
            parameters=parameters,
        )

    def _last_candidate_query_count(self, query: str) -> int:
        if not self.trace_events:
            return 1
        event = self.trace_events[-1]
        if event.get("stage") == "rag.merge_candidates" and event.get("query") == query:
            return int(event.get("candidate_queries", 1))
        return 1

    def _retrieve_once(
        self,
        *,
        query: str,
        namespace: str,
        limit: int,
        score_threshold: float | None,
    ) -> list[dict[str, Any]]:
        query_vector = self._encode_one(query)
        hits = self.vector_store.search_similar(
            query_vector=query_vector,
            limit=limit,
            score_threshold=score_threshold,
            where={
                "namespace": namespace,
                "memory_type": "rag_chunk",
                "data_source": "my_rag_tool",
            },
        )
        results: list[dict[str, Any]] = []
        seen_contents: set[str] = set()
        for hit in hits:
            score = float(hit.get("score", 0.0))
            if score_threshold is not None and score < score_threshold:
                continue
            meta = hit.get("metadata", {}) or {}
            content = str(meta.get("content", ""))
            if not content or content in seen_contents:
                continue
            seen_contents.add(content)
            results.append(
                {
                    "id": hit.get("id"),
                    "score": score,
                    "metadata": meta,
                }
            )
        return results[:limit]

    def _build_candidate_queries(self, query: str, parameters: dict[str, Any]) -> list[str]:
        queries = [query]
        if self._as_bool(parameters.get("enable_mqe")):
            queries.extend(self._generate_mqe_queries(query))
        if self._as_bool(parameters.get("enable_hyde")):
            queries.append(self._generate_hyde_document(query))

        deduped: list[str] = []
        seen: set[str] = set()
        for item in queries:
            normalized = item.strip()
            if normalized and normalized not in seen:
                deduped.append(normalized)
                seen.add(normalized)
        return deduped

    def _generate_mqe_queries(self, query: str) -> list[str]:
        prompt = [
            {
                "role": "system",
                "content": "你是一个检索查询改写器，只输出改写后的查询，每行一个。",
            },
            {
                "role": "user",
                "content": f"请把下面问题改写成 3 个适合向量检索的中文查询：\n{query}",
            },
        ]
        response = str(self.llm.invoke(prompt)).strip()
        queries = [
            line.strip("- 0123456789.、\t ")
            for line in response.splitlines()
            if line.strip()
        ]
        self.trace_events.append(
            {
                "stage": "rag.expand_mqe",
                "query": query,
                "generated": queries,
            }
        )
        return queries

    def _generate_hyde_document(self, query: str) -> str:
        prompt = [
            {
                "role": "system",
                "content": "你是一个 HyDE 检索助手，只生成假设性答案文本。",
            },
            {
                "role": "user",
                "content": f"请针对问题生成一段可能出现在知识库中的假设性答案：\n{query}",
            },
        ]
        document = str(self.llm.invoke(prompt)).strip()
        self.trace_events.append(
            {
                "stage": "rag.expand_hyde",
                "query": query,
                "document": document,
            }
        )
        return document

    def _merge_candidates(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        by_key: dict[str, dict[str, Any]] = {}
        for candidate in candidates:
            meta = candidate.get("metadata", {}) or {}
            key = str(candidate.get("id") or meta.get("content") or "")
            if not key:
                continue
            existing = by_key.get(key)
            if existing is None or candidate["score"] > existing["score"]:
                by_key[key] = candidate
        merged = list(by_key.values())
        merged.sort(key=lambda item: item["score"], reverse=True)
        return merged

    def _maybe_keyword_rerank(
        self,
        *,
        query: str,
        results: list[dict[str, Any]],
        limit: int,
        parameters: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if not self._as_bool(parameters.get("enable_keyword_rerank")):
            return results[:limit]
        weight = float(parameters.get("keyword_weight", 0.2))
        reranked = self._keyword_rerank(query=query, results=results, weight=weight)
        self.trace_events.append(
            {
                "stage": "rag.keyword_rerank",
                "query": query,
                "candidates": len(results),
                "keyword_weight": weight,
            }
        )
        return reranked[:limit]

    def _keyword_rerank(
        self,
        *,
        query: str,
        results: list[dict[str, Any]],
        weight: float,
    ) -> list[dict[str, Any]]:
        query_terms = self._keyword_terms(query)
        reranked: list[dict[str, Any]] = []
        for item in results:
            meta = item.get("metadata", {}) or {}
            content = str(meta.get("content", ""))
            keyword_score = float(len(query_terms & self._keyword_terms(content)))
            vector_score = float(item.get("vector_score", item.get("score", 0.0)))
            reranked.append(
                {
                    **item,
                    "vector_score": vector_score,
                    "keyword_score": keyword_score,
                    "score": vector_score + keyword_score * weight,
                }
            )
        reranked.sort(
            key=lambda item: (
                item.get("score", 0.0),
                item.get("keyword_score", 0.0),
                item.get("vector_score", 0.0),
            ),
            reverse=True,
        )
        return reranked

    def _keyword_terms(self, text: str) -> set[str]:
        """Extract explainable rerank terms from mixed Chinese/English text.

        这个重排不是替代 embedding，而是补足业务检索里常见的精确标识符：
        发票号、订单号、金额字段等一旦命中，就应该在向量候选中更靠前。
        """

        normalized = text.lower()
        raw_terms = re.findall(r"[a-z0-9][a-z0-9_-]*|[\u4e00-\u9fff]+", normalized)
        terms: set[str] = set()
        for term in raw_terms:
            if not term:
                continue
            terms.add(term)
            if re.fullmatch(r"[\u4e00-\u9fff]+", term):
                terms.update(term[index : index + 2] for index in range(max(0, len(term) - 1)))
        return terms

    def _as_bool(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)

    def _chunk_text(self, text: str, *, chunk_size: int, chunk_overlap: int) -> list[str]:
        return [
            chunk["content"]
            for chunk in self._build_chunks(
                text,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        ]

    def _build_chunks(
        self,
        text: str,
        *,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[dict[str, Any]]:
        chunk_size = max(1, chunk_size)
        chunk_overlap = max(0, min(chunk_overlap, chunk_size - 1))
        units = self._split_semantic_units(text)
        if not units:
            return []

        chunks: list[dict[str, Any]] = []
        current_parts: list[str] = []
        current_start: int | None = None
        current_end = 0
        current_section: str | None = None

        for unit in units:
            content = str(unit["content"]).strip()
            if not content:
                continue
            if unit.get("is_heading") and current_parts:
                chunks.append(
                    self._make_chunk(
                        current_parts,
                        start_char=current_start or 0,
                        end_char=current_end,
                        section_title=current_section,
                    )
                )
                current_parts = []
                current_start = None

            if len(content) > chunk_size:
                if current_parts:
                    chunks.append(
                        self._make_chunk(
                            current_parts,
                            start_char=current_start or 0,
                            end_char=current_end,
                            section_title=current_section,
                        )
                    )
                    current_parts = []
                    current_start = None
                chunks.extend(
                    self._split_long_unit(
                        content,
                        start_char=int(unit["start_char"]),
                        section_title=unit.get("section_title"),
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )
                )
                current_end = int(unit["end_char"])
                current_section = unit.get("section_title")
                continue

            joined = "\n\n".join([*current_parts, content]) if current_parts else content
            if current_parts and len(joined) > chunk_size:
                if self._can_attach_to_heading(current_parts, content, chunk_size):
                    current_parts.append(content)
                    current_end = int(unit["end_char"])
                    current_section = unit.get("section_title") or current_section
                    continue
                chunks.append(
                    self._make_chunk(
                        current_parts,
                        start_char=current_start or 0,
                        end_char=current_end,
                        section_title=current_section,
                    )
                )
                # 段落/标题之间保持语义边界，不把上一块尾巴塞进下一块。
                # 字符级 overlap 只用于同一个超长语义单元，避免半截标题或半截句子。
                current_parts = [content]
                current_start = int(unit["start_char"])
            else:
                current_parts.append(content)
                if current_start is None:
                    current_start = int(unit["start_char"])
            current_end = int(unit["end_char"])
            current_section = unit.get("section_title") or current_section

        if current_parts:
            chunks.append(
                self._make_chunk(
                    current_parts,
                    start_char=current_start or 0,
                    end_char=current_end,
                    section_title=current_section,
                )
            )
        return chunks

    def _can_attach_to_heading(
        self,
        current_parts: list[str],
        content: str,
        chunk_size: int,
    ) -> bool:
        """Keep a Markdown heading with its first paragraph when slightly oversized."""

        if len(current_parts) != 1:
            return False
        if not current_parts[0].lstrip().startswith("#"):
            return False
        if content.lstrip().startswith("#"):
            return False
        joined = "\n\n".join([current_parts[0], content])
        return len(joined) <= max(chunk_size + 80, int(chunk_size * 2))

    def _split_semantic_units(self, text: str) -> list[dict[str, Any]]:
        """Split text into heading/paragraph aware units before vectorization."""

        units: list[dict[str, Any]] = []
        current_section: str | None = None
        position = 0
        for block in self._iter_text_blocks(text):
            content = block["content"]
            stripped = content.strip()
            if stripped.startswith("#"):
                current_section = stripped.lstrip("#").strip() or current_section
            units.append(
                {
                    "content": stripped,
                    "start_char": block["start_char"],
                    "end_char": block["end_char"],
                    "section_title": current_section,
                    "is_heading": stripped.startswith("#"),
                }
            )
            position = int(block["end_char"])
        if units:
            return units

        stripped = text.strip()
        return (
            [
                {
                    "content": stripped,
                    "start_char": 0,
                    "end_char": len(text),
                    "section_title": None,
                    "is_heading": False,
                }
            ]
            if stripped
            else []
        )

    def _iter_text_blocks(self, text: str) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []
        cursor = 0
        for raw_block in text.split("\n\n"):
            start = cursor
            end = start + len(raw_block)
            cursor = end + 2
            stripped = raw_block.strip()
            if not stripped:
                continue
            leading = len(raw_block) - len(raw_block.lstrip())
            trailing = len(raw_block.rstrip())
            blocks.append(
                {
                    "content": stripped,
                    "start_char": start + leading,
                    "end_char": start + trailing,
                }
            )
        return blocks

    def _split_long_unit(
        self,
        content: str,
        *,
        start_char: int,
        section_title: str | None,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[dict[str, Any]]:
        chunks: list[dict[str, Any]] = []
        start = 0
        while start < len(content):
            end = min(len(content), start + chunk_size)
            chunk = content[start:end].strip()
            if chunk:
                chunks.append(
                    {
                        "content": chunk,
                        "start_char": start_char + start,
                        "end_char": start_char + end,
                        "section_title": section_title,
                    }
                )
            if end >= len(content):
                break
            start = end - chunk_overlap
        return chunks

    def _make_chunk(
        self,
        parts: list[str],
        *,
        start_char: int,
        end_char: int,
        section_title: str | None,
    ) -> dict[str, Any]:
        return {
            "content": "\n\n".join(part for part in parts if part).strip(),
            "start_char": start_char,
            "end_char": end_char,
            "section_title": section_title,
        }

    def _tail_overlap(self, text: str, chunk_overlap: int) -> str:
        if chunk_overlap <= 0:
            return ""
        return text[-chunk_overlap:].strip()

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
