from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv
from hello_agents import HelloAgentsLLM

ROOT = Path(__file__).resolve().parents[1]
from app.my_memory_system import (
    EpisodicMemoryStore,
    MemoryRAGDashboard,
    MyMemoryManager,
    MyRAGTool,
    PerceptualMemoryStore,
    SemanticMemoryStore,
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
    external_python = ROOT / ".venv-asr" / "Scripts" / "python.exe"
    external_worker = ROOT / "scripts" / "chapter8_multimodal_worker.py"
    use_external_runtime = external_python.exists() and external_worker.exists()
    perception_tool = build_multimodal_perception_tool(
        manager=manager,
        rag_tool=rag_tool,
        rag_namespace="business_multimodal",
        image_ocr=image_ocr,
        audio_asr=audio_asr,
        enable_ocr=True,
        enable_asr=True,
        enable_image_embedding=False,
        enable_audio_embedding=False,
        model_root=ROOT / "models",
        multimodal_python=external_python if use_external_runtime else None,
        multimodal_worker=external_worker if use_external_runtime else None,
        collection_name="chapter8_dashboard_multimodal",
    )
    return MemoryRAGDashboard(
        perception_tool=perception_tool,
        rag_tool=rag_tool,
        memory_manager=manager,
        rag_namespace="business_multimodal",
    )


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
