import hashlib
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_NAMES = {".env", ".venv", "__pycache__", ".git"}
TEXT_SUFFIXES = {".md", ".py", ".txt", ".json", ".json5", ".toml", ".yaml", ".yml"}

_VECTOR_STORE: InMemoryVectorStore | None = None
_CHUNKS: list[Document] | None = None
_RERANKER = None


@dataclass
class ScoredDocument:
    document: Document
    score: float
    reason: str


class LocalHashEmbeddings(Embeddings):
    """Small local embedding fallback for demo RAG when remote embeddings are unavailable."""

    def __init__(self, dimensions: int = 768):
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in _tokenize(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]


class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def _tokenize(text: str) -> list[str]:
    text = text.lower()
    latin_tokens = re.findall(r"[a-z0-9_./:-]+", text)
    cjk_tokens = re.findall(r"[\u4e00-\u9fff]", text)
    return latin_tokens + cjk_tokens


def _is_excluded(path: Path) -> bool:
    try:
        parts = path.relative_to(PROJECT_ROOT).parts
    except ValueError:
        return True
    return any(part in EXCLUDED_NAMES for part in parts)


def _iter_text_files() -> list[Path]:
    files = []
    for path in PROJECT_ROOT.rglob("*"):
        if _is_excluded(path) or not path.is_file():
            continue
        if path.suffix.lower() in TEXT_SUFFIXES:
            files.append(path)
    return sorted(files)


def _load_documents() -> list[Document]:
    documents = []
    for path in _iter_text_files():
        source = str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
        text = path.read_text(encoding="utf-8", errors="replace")
        if text.strip():
            documents.append(Document(page_content=text, metadata={"source": source}))
    return documents


def _split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    for index, chunk in enumerate(chunks, start=1):
        chunk.metadata["chunk_id"] = index
    return chunks


def _get_embeddings() -> Embeddings:
    settings = get_settings()
    if settings.embedding_provider == "openai":
        return OpenAIEmbeddings(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            model=settings.embedding_model,
        )
    if settings.embedding_provider == "sentence_transformers":
        return SentenceTransformerEmbeddings(settings.local_embedding_model)
    return LocalHashEmbeddings()


def build_vector_store(force_rebuild: bool = False) -> InMemoryVectorStore:
    global _CHUNKS, _VECTOR_STORE
    if _VECTOR_STORE is not None and not force_rebuild:
        return _VECTOR_STORE

    documents = _split_documents(_load_documents())
    _CHUNKS = documents
    embeddings = _get_embeddings()
    _VECTOR_STORE = InMemoryVectorStore.from_documents(documents, embeddings)
    return _VECTOR_STORE


def _get_chunks() -> list[Document]:
    if _CHUNKS is None:
        build_vector_store()
    return _CHUNKS or []


def _rewrite_queries(query: str) -> list[str]:
    settings = get_settings()
    if not settings.rag_query_rewrite:
        return [query]

    llm = ChatOpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        model=settings.model_name,
        temperature=0,
    )
    try:
        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "Rewrite the user's question into up to 3 concise search queries. "
                        "Return one query per line. Do not add explanations."
                    )
                ),
                HumanMessage(content=query),
            ]
        )
    except Exception:
        return [query]

    rewrites = [line.strip("- 0123456789.").strip() for line in response.content.splitlines()]
    queries = [query]
    for rewritten in rewrites:
        if rewritten and rewritten not in queries:
            queries.append(rewritten)
    return queries[:4]


def _dense_search(queries: list[str], top_k: int) -> list[ScoredDocument]:
    vector_store = build_vector_store()
    results: list[ScoredDocument] = []
    seen = set()
    for query in queries:
        for document, score in vector_store.similarity_search_with_score(query, k=top_k):
            key = _doc_key(document)
            if key in seen:
                continue
            seen.add(key)
            results.append(ScoredDocument(document=document, score=float(score), reason="dense"))
    return results


def _sparse_search(queries: list[str], top_k: int) -> list[ScoredDocument]:
    chunks = _get_chunks()
    tokenized = [Counter(_tokenize(chunk.page_content + " " + chunk.metadata["source"])) for chunk in chunks]
    doc_freq = Counter()
    for terms in tokenized:
        doc_freq.update(terms.keys())

    avg_len = sum(sum(terms.values()) for terms in tokenized) / max(1, len(tokenized))
    results: dict[str, ScoredDocument] = {}
    for query in queries:
        query_terms = Counter(_tokenize(query))
        scored = []
        for chunk, terms in zip(chunks, tokenized):
            score = _bm25_score(query_terms, terms, doc_freq, len(chunks), avg_len)
            if score > 0:
                scored.append((chunk, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        for chunk, score in scored[:top_k]:
            key = _doc_key(chunk)
            previous = results.get(key)
            if previous is None or score > previous.score:
                results[key] = ScoredDocument(document=chunk, score=score, reason="sparse")
    return list(results.values())


def _bm25_score(
    query_terms: Counter[str],
    doc_terms: Counter[str],
    doc_freq: Counter[str],
    total_docs: int,
    avg_len: float,
) -> float:
    k1 = 1.5
    b = 0.75
    doc_len = sum(doc_terms.values())
    score = 0.0
    for term, query_count in query_terms.items():
        tf = doc_terms.get(term, 0)
        if tf == 0:
            continue
        idf = math.log(1 + (total_docs - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
        denom = tf + k1 * (1 - b + b * doc_len / max(avg_len, 1))
        score += query_count * idf * (tf * (k1 + 1)) / denom
    return score


def _rrf_fusion(result_sets: list[list[ScoredDocument]], limit: int) -> list[ScoredDocument]:
    k = 60
    fused_scores: defaultdict[str, float] = defaultdict(float)
    documents: dict[str, Document] = {}
    reasons: defaultdict[str, list[str]] = defaultdict(list)

    for result_set in result_sets:
        for rank, result in enumerate(result_set, start=1):
            key = _doc_key(result.document)
            fused_scores[key] += 1 / (k + rank)
            documents[key] = result.document
            if result.reason not in reasons[key]:
                reasons[key].append(result.reason)

    ranked = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
    return [
        ScoredDocument(documents[key], score, "+".join(reasons[key]))
        for key, score in ranked[:limit]
    ]


def _expand_neighbor_chunks(results: list[ScoredDocument]) -> list[ScoredDocument]:
    chunks = _get_chunks()
    by_source: defaultdict[str, list[Document]] = defaultdict(list)
    for chunk in chunks:
        by_source[chunk.metadata["source"]].append(chunk)

    expanded: dict[str, ScoredDocument] = {_doc_key(result.document): result for result in results}
    for result in results:
        source = result.document.metadata["source"]
        chunk_id = result.document.metadata["chunk_id"]
        for neighbor in by_source[source]:
            if abs(neighbor.metadata["chunk_id"] - chunk_id) <= 1:
                key = _doc_key(neighbor)
                expanded.setdefault(
                    key,
                    ScoredDocument(neighbor, result.score * 0.9, result.reason + "+window"),
                )
    return sorted(expanded.values(), key=lambda item: item.score, reverse=True)


def _rerank(query: str, results: list[ScoredDocument], top_k: int) -> list[ScoredDocument]:
    settings = get_settings()
    if not settings.rag_reranker_enabled or not results:
        return results[:top_k]

    if settings.rag_reranker_provider == "llm":
        return _llm_rerank(query, results, top_k)

    reranker = _get_reranker(settings.rag_reranker_model)
    pairs = [[query, result.document.page_content] for result in results]
    try:
        scores = reranker.predict(pairs)
    except Exception:
        return results[:top_k]

    reranked = [
        ScoredDocument(result.document, float(score), result.reason + "+rerank")
        for result, score in zip(results, scores)
    ]
    reranked.sort(key=lambda item: item.score, reverse=True)
    return reranked[:top_k]


def _llm_rerank(query: str, results: list[ScoredDocument], top_k: int) -> list[ScoredDocument]:
    candidates = results[:12]
    candidate_text = []
    for index, result in enumerate(candidates, start=1):
        source = _doc_key(result.document)
        content = result.document.page_content[:900].replace("\n", " ")
        candidate_text.append(f"[{index}] {source}\n{content}")

    settings = get_settings()
    llm = ChatOpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        model=settings.model_name,
        temperature=0,
    )
    try:
        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a retrieval reranker. Rank the candidate passages by "
                        "usefulness for answering the query. Return only candidate numbers "
                        "in best-to-worst order, separated by commas."
                    )
                ),
                HumanMessage(
                    content=f"Query: {query}\n\nCandidates:\n\n" + "\n\n".join(candidate_text)
                ),
            ]
        )
    except Exception:
        return results[:top_k]

    order = [int(match) for match in re.findall(r"\d+", response.content)]
    ordered = []
    used = set()
    for rank, candidate_number in enumerate(order, start=1):
        index = candidate_number - 1
        if 0 <= index < len(candidates) and index not in used:
            used.add(index)
            original = candidates[index]
            ordered.append(
                ScoredDocument(
                    original.document,
                    1.0 / rank,
                    original.reason + "+llm_rerank",
                )
            )

    for index, original in enumerate(candidates):
        if index not in used:
            ordered.append(original)

    return ordered[:top_k]


def _get_reranker(model_name: str):
    global _RERANKER
    if _RERANKER is None:
        from sentence_transformers import CrossEncoder

        _RERANKER = CrossEncoder(model_name)
    return _RERANKER


def _doc_key(document: Document) -> str:
    return f"{document.metadata.get('source')}#{document.metadata.get('chunk_id')}"


def search_documents(query: str, top_k: int = 5) -> list[ScoredDocument]:
    settings = get_settings()
    final_top_k = max(1, min(top_k or settings.rag_final_top_k, 10))
    queries = _rewrite_queries(query)
    dense = _dense_search(queries, settings.rag_dense_top_k)
    sparse = _sparse_search(queries, settings.rag_sparse_top_k)
    fused = _rrf_fusion([dense, sparse], limit=max(settings.rag_final_top_k * 4, 20))
    expanded = _expand_neighbor_chunks(fused)
    return _rerank(query, expanded, final_top_k)


def format_search_results(query: str, top_k: int = 5) -> str:
    results = search_documents(query=query, top_k=top_k)
    if not results:
        return "No relevant project documents found."

    blocks = []
    for result in results:
        document = result.document
        source = document.metadata.get("source", "unknown")
        chunk_id = document.metadata.get("chunk_id", "?")
        blocks.append(
            "\n".join(
                [
                    f"Source: {source}#chunk-{chunk_id}",
                    f"Score: {result.score:.3f}",
                    f"Retrieval: {result.reason}",
                    "Content:",
                    document.page_content,
                ]
            )
        )
    return "\n\n---\n\n".join(blocks)
