from __future__ import annotations

import argparse

from app.my_memory_system import MemoryRAGDashboard, build_memory_rag_dashboard_app
from scripts.chapter8_business_multimodal_demo import build_business_multimodal_demo


def build_dashboard_demo() -> MemoryRAGDashboard:
    """Build a ready-to-use dashboard backed by the business multimodal demo stack.

    这个入口面向页面手工体验：OCR/ASR 使用可控的注入函数，RAG 和记忆
    使用内存实现，因此不用先启动 Qdrant、Neo4j 或真实多模态模型。
    """

    perception_tool, rag_tool, manager = build_business_multimodal_demo()
    return MemoryRAGDashboard(
        perception_tool=perception_tool,
        rag_tool=rag_tool,
        memory_manager=manager,
        rag_namespace="business_multimodal",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch chapter 8 Memory/RAG dashboard demo.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7868)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    dashboard = build_dashboard_demo()
    app = build_memory_rag_dashboard_app(dashboard)
    app.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
