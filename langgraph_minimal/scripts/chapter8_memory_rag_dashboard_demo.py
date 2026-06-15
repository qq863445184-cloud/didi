from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv
from hello_agents import HelloAgentsLLM

ROOT = Path(__file__).resolve().parents[1]
from app.my_memory_system import (
    DocumentParserPipeline,
    EpisodicMemoryStore,
    MemoryRAGDashboard,
    MyMemoryManager,
    MyRAGTool,
    PerceptualMemoryStore,
    SemanticMemoryStore,
    SQLiteDocumentStore,
    WorkingMemoryStore,
    build_memory_rag_dashboard_app,
    build_multimodal_perception_tool,
)
from scripts.chapter8_business_multimodal_demo import (
    BusinessDemoLLM,
    BusinessKeywordEmbedder,
    InMemoryGraphStore,
    InMemoryVectorStore,
    build_business_multimodal_demo,
)


def build_dashboard_demo(
    *,
    prefer_real_llm: bool = False,
    prefer_real_multimodal: bool = False,
    image_ocr: Callable[[Path], str] | None = None,
    audio_asr: Callable[[Path], str] | None = None,
) -> MemoryRAGDashboard:
    """Build a ready-to-use dashboard backed by the business multimodal demo stack.

    这个入口面向页面手工体验：OCR/ASR 使用可控的注入函数，RAG 和记忆
    使用内存实现，因此不用先启动 Qdrant、Neo4j。
    当 prefer_real_llm=True 时，RAG 的 ask 阶段会优先接入 .env 中的真实大模型。
    当 prefer_real_multimodal=True 时，图片/音频会通过真实 OCR/ASR 处理器入库。
    """

    llm = None
    llm_mode = "demo"
    if prefer_real_llm:
        llm = _build_real_llm()
        llm_mode = "real"

    if prefer_real_multimodal:
        return _build_real_multimodal_dashboard(
            llm=llm,
            llm_mode=llm_mode,
            image_ocr=image_ocr,
            audio_asr=audio_asr,
        )

    perception_tool, rag_tool, manager = build_business_multimodal_demo(
        llm=llm,
        llm_mode=llm_mode,
    )
    return MemoryRAGDashboard(
        perception_tool=perception_tool,
        rag_tool=rag_tool,
        memory_manager=manager,
        rag_namespace="business_multimodal",
    )


def build_persistent_dashboard_demo(
    *,
    data_dir: str | Path | None = None,
    prefer_real_llm: bool = False,
    prefer_real_multimodal: bool = True,
    prefer_external_backends: bool = False,
    strict_backends: bool = False,
    image_ocr: Callable[[Path], str] | None = None,
    audio_asr: Callable[[Path], str] | None = None,
) -> MemoryRAGDashboard:
    """Build a Chapter 8 dashboard with persistent local metadata.

    本地持久化层始终启用：RAG 文档/chunk 元数据写入 SQLite，working 和
    perceptual 记忆写入 JSON。Qdrant/Neo4j 属于外部后端，只有在
    prefer_external_backends=True 时接入；如果外部服务不可用，非 strict 模式会
    回退到内存向量/图存储，让页面仍然可用。
    """

    load_dotenv(ROOT / ".env", override=True)
    workspace = Path(data_dir or ROOT / "memory_data" / "persistent_dashboard")
    workspace.mkdir(parents=True, exist_ok=True)

    llm = None
    llm_mode = "demo"
    if prefer_real_llm:
        llm = _build_real_llm()
        llm_mode = "real"

    embedder, rag_vector_store, semantic_vector_store, graph_store, backend_mode = (
        _build_persistent_backends(
            prefer_external_backends=prefer_external_backends,
            strict_backends=strict_backends,
        )
    )
    rag_tool = MyRAGTool(
        embedder=embedder,
        vector_store=rag_vector_store,
        llm=llm or BusinessDemoLLM(),
        document_store=SQLiteDocumentStore(workspace / "rag_documents.sqlite3"),
        parser_pipeline=_build_document_parser_pipeline(
            prefer_real_multimodal=prefer_real_multimodal,
            image_ocr=image_ocr,
            audio_asr=audio_asr,
        ),
        collection_name="chapter8_persistent_rag",
    )
    rag_tool.llm_mode = llm_mode
    rag_tool.backend_mode = backend_mode

    manager = MyMemoryManager(
        user_id="dashboard_persistent_user",
        stores={
            "working": WorkingMemoryStore(
                persistence_path=str(workspace / "working_memory.json")
            ),
            "semantic": SemanticMemoryStore(
                embedder=embedder,
                vector_store=semantic_vector_store,
                graph_store=graph_store,
                collection_name="chapter8_persistent_semantic",
                restore_existing=prefer_external_backends,
            ),
            "episodic": EpisodicMemoryStore(
                graph_store=graph_store,
                restore_existing=prefer_external_backends,
            ),
            "perceptual": PerceptualMemoryStore(
                persistence_path=str(workspace / "perceptual_memory.json")
            ),
        },
    )

    perception_tool = build_multimodal_perception_tool(
        manager=manager,
        rag_tool=rag_tool,
        rag_namespace="business_multimodal",
        image_ocr=image_ocr or (_real_image_ocr if prefer_real_multimodal else None),
        audio_asr=audio_asr or (_real_audio_asr if prefer_real_multimodal else None),
        enable_ocr=prefer_real_multimodal,
        enable_asr=prefer_real_multimodal,
        enable_image_embedding=False,
        enable_audio_embedding=False,
        model_root=ROOT / "models",
        multimodal_worker=_external_multimodal_worker(),
        external_timeout_seconds=_dashboard_multimodal_timeout(),
        collection_name="chapter8_persistent_perceptual",
    )
    return MemoryRAGDashboard(
        perception_tool=perception_tool,
        rag_tool=rag_tool,
        memory_manager=manager,
        rag_namespace="business_multimodal",
    )


def _build_real_multimodal_dashboard(
    *,
    llm,
    llm_mode: str,
    image_ocr: Callable[[Path], str] | None,
    audio_asr: Callable[[Path], str] | None,
) -> MemoryRAGDashboard:
    """Build the browser-facing stack that extracts real image/audio content."""

    embedder = BusinessKeywordEmbedder()
    rag_tool = MyRAGTool(
        embedder=embedder,
        vector_store=InMemoryVectorStore(),
        llm=llm or BusinessDemoLLM(),
        collection_name="chapter8_dashboard_multimodal",
    )
    rag_tool.llm_mode = llm_mode
    manager = MyMemoryManager(
        user_id="dashboard_multimodal_user",
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
    external_worker = ROOT / "scripts" / "chapter8_multimodal_worker.py"
    use_external_runtime = external_worker.exists()
    perception_tool = build_multimodal_perception_tool(
        manager=manager,
        rag_tool=rag_tool,
        rag_namespace="business_multimodal",
        image_ocr=image_ocr or _real_image_ocr,
        audio_asr=audio_asr or _real_audio_asr,
        enable_ocr=True,
        enable_asr=True,
        enable_image_embedding=False,
        enable_audio_embedding=False,
        model_root=ROOT / "models",
        multimodal_worker=external_worker if use_external_runtime else None,
        external_timeout_seconds=_dashboard_multimodal_timeout(),
        collection_name="chapter8_dashboard_multimodal",
    )
    return MemoryRAGDashboard(
        perception_tool=perception_tool,
        rag_tool=rag_tool,
        memory_manager=manager,
        rag_namespace="business_multimodal",
    )


def _build_persistent_backends(
    *,
    prefer_external_backends: bool,
    strict_backends: bool,
) -> tuple[object, object, object, object, str]:
    embedder = BusinessKeywordEmbedder()
    rag_vector_store = InMemoryVectorStore()
    semantic_vector_store = InMemoryVectorStore()
    graph_store = InMemoryGraphStore()
    backend_mode = "local_persistent"

    if not prefer_external_backends:
        return embedder, rag_vector_store, semantic_vector_store, graph_store, backend_mode

    try:
        from hello_agents.memory.embedding import get_text_embedder
        from hello_agents.memory.storage.neo4j_store import Neo4jGraphStore
        from hello_agents.memory.storage.qdrant_store import QdrantVectorStore

        embedder = get_text_embedder()
        rag_vector_store = QdrantVectorStore(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY") or None,
            collection_name="chapter8_persistent_rag",
            vector_size=int(os.getenv("EMBED_VECTOR_SIZE", "384")),
            distance="cosine",
        )
        semantic_vector_store = QdrantVectorStore(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY") or None,
            collection_name="chapter8_persistent_semantic",
            vector_size=int(os.getenv("EMBED_VECTOR_SIZE", "384")),
            distance="cosine",
        )
        graph_store = Neo4jGraphStore(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            username=os.getenv("NEO4J_USERNAME", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "hello-agents-password"),
            database=os.getenv("NEO4J_DATABASE", "neo4j"),
        )
        backend_mode = "qdrant_neo4j_sqlite"
        return embedder, rag_vector_store, semantic_vector_store, graph_store, backend_mode
    except Exception:
        if strict_backends:
            raise
        return embedder, rag_vector_store, semantic_vector_store, graph_store, backend_mode


def _build_document_parser_pipeline(
    *,
    prefer_real_multimodal: bool,
    image_ocr: Callable[[Path], str] | None,
    audio_asr: Callable[[Path], str] | None,
) -> DocumentParserPipeline:
    if not prefer_real_multimodal:
        return DocumentParserPipeline(image_ocr=image_ocr, audio_asr=audio_asr)
    return DocumentParserPipeline(
        image_ocr=image_ocr or _real_image_ocr,
        audio_asr=audio_asr or _real_audio_asr,
    )


def _external_multimodal_python() -> Path | None:
    return _external_runtime_python(
        "CHAPTER8_DASHBOARD_MULTIMODAL_PYTHON",
        ROOT / ".venv-asr" / "Scripts" / "python.exe",
    )


def _external_ocr_python() -> Path | None:
    return _external_runtime_python(
        "CHAPTER8_DASHBOARD_OCR_PYTHON",
        _external_multimodal_python(),
    )


def _external_asr_python() -> Path | None:
    return _external_runtime_python(
        "CHAPTER8_DASHBOARD_ASR_PYTHON",
        _external_multimodal_python(),
    )


def _external_runtime_python(env_name: str, default_path: Path | None) -> Path | None:
    raw_path = os.getenv(env_name)
    path = Path(raw_path).expanduser() if raw_path else default_path
    return path if path is not None and path.exists() else None


def _external_multimodal_worker() -> Path | None:
    path = ROOT / "scripts" / "chapter8_multimodal_worker.py"
    return path if path.exists() else None


def _dashboard_multimodal_timeout() -> float:
    """Timeout for browser-triggered OCR/ASR jobs.

    上传页面会把真实 OCR/ASR 放到后台执行。这里仍保留超时，避免单个
    worker 长时间占用资源；需要测试冷启动重模型时可通过环境变量调大。
    """

    return float(os.getenv("CHAPTER8_DASHBOARD_MULTIMODAL_TIMEOUT", "180"))


def _real_image_ocr(path: Path) -> str:
    from app.my_memory_system.multimodal_pipeline import ExternalRuntimeOCR, PaddleOCRVLOCR

    external_python = _external_ocr_python()
    external_worker = _external_multimodal_worker()
    if external_python is not None and external_worker is not None:
        processor = ExternalRuntimeOCR(
            python_path=external_python,
            worker_path=external_worker,
            model_dir=ROOT / "models" / "PaddlePaddle" / "PaddleOCR-VL-1.6",
            timeout_seconds=_dashboard_multimodal_timeout(),
        )
    else:
        processor = PaddleOCRVLOCR(
            model_dir=ROOT / "models" / "PaddlePaddle" / "PaddleOCR-VL-1.6"
        )
    return str(processor(path))


def _real_audio_asr(path: Path) -> str:
    from app.my_memory_system.multimodal_pipeline import ExternalRuntimeASR, SenseVoiceASR

    external_python = _external_asr_python()
    external_worker = _external_multimodal_worker()
    if external_python is not None and external_worker is not None:
        processor = ExternalRuntimeASR(
            python_path=external_python,
            worker_path=external_worker,
            model_dir=ROOT / "models" / "iic" / "SenseVoiceSmall",
            timeout_seconds=_dashboard_multimodal_timeout(),
        )
    else:
        processor = SenseVoiceASR(model_dir=ROOT / "models" / "iic" / "SenseVoiceSmall")
    return str(processor(path))


def _build_real_llm() -> HelloAgentsLLM:
    """Create the real generation model used by the browser dashboard."""

    load_dotenv(ROOT / ".env", override=True)
    model = os.getenv("LLM_MODEL_ID") or os.getenv("OPENAI_MODEL") or os.getenv("MODEL_NAME")
    api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    if not model or not api_key or not base_url:
        raise RuntimeError(
            "真实生成模式需要在 .env 中配置 LLM_MODEL_ID/LLM_API_KEY/LLM_BASE_URL "
            "或 OPENAI_MODEL/OPENAI_API_KEY/OPENAI_BASE_URL。"
        )
    return HelloAgentsLLM(
        provider=os.getenv("LLM_PROVIDER", "openai"),
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=float(os.getenv("RAG_LLM_TEMPERATURE", "0.2")),
        max_tokens=int(os.getenv("RAG_LLM_MAX_TOKENS", "1200")),
        timeout=int(os.getenv("RAG_LLM_TIMEOUT", "120")),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch chapter 8 Memory/RAG dashboard demo.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7868)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--real-llm", action="store_true", help="Use .env LLM config for RAG generation.")
    parser.add_argument("--real-multimodal", action="store_true", help="Use OCR/ASR for uploaded image/audio files.")
    args = parser.parse_args()

    dashboard = build_dashboard_demo(
        prefer_real_llm=args.real_llm,
        prefer_real_multimodal=args.real_multimodal,
    )
    app = build_memory_rag_dashboard_app(dashboard)
    app.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
