import os
import sys
from dataclasses import dataclass

from dotenv import load_dotenv


class ConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class AgentSettings:
    openai_api_key: str
    openai_base_url: str
    model_name: str
    embedding_provider: str
    embedding_model: str
    local_embedding_model: str
    rag_query_rewrite: bool
    rag_dense_top_k: int
    rag_sparse_top_k: int
    rag_final_top_k: int
    rag_reranker_enabled: bool
    rag_reranker_provider: str
    rag_reranker_model: str
    rag_verify_enabled: bool
    rag_max_attempts: int
    recursion_limit: int = 10


def configure_stdio() -> None:
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(errors="replace")
        sys.stdout.reconfigure(errors="replace")


def get_settings() -> AgentSettings:
    load_dotenv(override=True)

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    model_name = os.getenv("MODEL_NAME", "gpt-5.5").strip()
    embedding_provider = os.getenv("EMBEDDING_PROVIDER", "local_hash").strip()
    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small").strip()
    local_embedding_model = os.getenv(
        "LOCAL_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"
    ).strip()
    rag_query_rewrite = os.getenv("RAG_QUERY_REWRITE", "true").strip().lower() == "true"
    rag_dense_top_k = _read_int("RAG_DENSE_TOP_K", 20)
    rag_sparse_top_k = _read_int("RAG_SPARSE_TOP_K", 20)
    rag_final_top_k = _read_int("RAG_FINAL_TOP_K", 5)
    rag_reranker_enabled = (
        os.getenv("RAG_RERANKER_ENABLED", "true").strip().lower() == "true"
    )
    rag_reranker_provider = os.getenv("RAG_RERANKER_PROVIDER", "llm").strip()
    rag_reranker_model = os.getenv("RAG_RERANKER_MODEL", "BAAI/bge-reranker-base").strip()
    rag_verify_enabled = os.getenv("RAG_VERIFY_ENABLED", "false").strip().lower() == "true"
    rag_max_attempts = _read_int("RAG_MAX_ATTEMPTS", 2)
    recursion_limit_raw = os.getenv("RECURSION_LIMIT", "10").strip()

    missing = [
        name
        for name, value in {
            "OPENAI_API_KEY": api_key,
            "OPENAI_BASE_URL": base_url,
            "MODEL_NAME": model_name,
            "EMBEDDING_PROVIDER": embedding_provider,
            "EMBEDDING_MODEL": embedding_model,
            "LOCAL_EMBEDDING_MODEL": local_embedding_model,
            "RAG_RERANKER_MODEL": rag_reranker_model,
        }.items()
        if not value
    ]
    if missing:
        raise ConfigError(f"Missing required environment variables: {', '.join(missing)}")

    try:
        recursion_limit = int(recursion_limit_raw)
    except ValueError as exc:
        raise ConfigError("RECURSION_LIMIT must be an integer.") from exc

    if recursion_limit < 2:
        raise ConfigError("RECURSION_LIMIT must be at least 2.")

    return AgentSettings(
        openai_api_key=api_key,
        openai_base_url=base_url,
        model_name=model_name,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        local_embedding_model=local_embedding_model,
        rag_query_rewrite=rag_query_rewrite,
        rag_dense_top_k=rag_dense_top_k,
        rag_sparse_top_k=rag_sparse_top_k,
        rag_final_top_k=rag_final_top_k,
        rag_reranker_enabled=rag_reranker_enabled,
        rag_reranker_provider=rag_reranker_provider,
        rag_reranker_model=rag_reranker_model,
        rag_verify_enabled=rag_verify_enabled,
        rag_max_attempts=rag_max_attempts,
        recursion_limit=recursion_limit,
    )


def _read_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer.") from exc
    if value < 1:
        raise ConfigError(f"{name} must be at least 1.")
    return value
