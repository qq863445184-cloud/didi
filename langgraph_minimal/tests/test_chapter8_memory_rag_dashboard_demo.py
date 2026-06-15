from scripts.chapter8_memory_rag_dashboard_demo import build_dashboard_demo


def test_build_dashboard_demo_wires_business_multimodal_stack():
    dashboard = build_dashboard_demo()

    assert dashboard.rag_tool is not None
    assert dashboard.perception_tool is not None
    assert dashboard.memory_manager is not None
    assert dashboard.rag_namespace == "business_multimodal"
