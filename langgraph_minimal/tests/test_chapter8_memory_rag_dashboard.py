from app.my_memory_system.memory_rag_dashboard import MemoryRAGDashboard
from scripts.chapter8_business_multimodal_demo import build_business_multimodal_demo


def test_memory_rag_dashboard_drives_multimodal_rag_and_memory(tmp_path):
    invoice = tmp_path / "invoice_INV-2026-001.png"
    audio = tmp_path / "support_call_ORD-2026-778.wav"
    invoice.write_bytes(b"fake invoice image bytes")
    audio.write_bytes(b"fake support audio bytes")

    perception_tool, rag_tool, manager = build_business_multimodal_demo()
    dashboard = MemoryRAGDashboard(
        perception_tool=perception_tool,
        rag_tool=rag_tool,
        memory_manager=manager,
        rag_namespace="business_multimodal",
    )

    invoice_output = dashboard.ingest_file(
        str(invoice),
        description="Finance invoice screenshot.",
        importance=0.85,
    )
    audio_output = dashboard.ingest_file(
        str(audio),
        description="Customer support call recording.",
        importance=0.9,
    )
    answer_output = dashboard.ask("Which invoice is tied to the refund request?", limit=3)
    inventory_output = dashboard.memory_inventory()
    trace_output = dashboard.trace()

    assert "Perceptual memory saved" in invoice_output
    assert "Perceptual memory saved" in audio_output
    assert "INV-2026-001" in answer_output
    assert "检索证据" in answer_output
    assert "perceptual" in inventory_output
    assert "semantic" in inventory_output
    assert "perception.sync_rag" in trace_output
