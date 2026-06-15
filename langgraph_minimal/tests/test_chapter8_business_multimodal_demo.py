from scripts.chapter8_business_multimodal_demo import run_business_multimodal_demo


def test_business_multimodal_demo_ingests_invoice_and_support_call(tmp_path):
    result = run_business_multimodal_demo(tmp_path)

    assert result["perceptual_count"] == 2
    assert result["semantic_count"] == 2
    assert result["episodic_count"] == 2
    assert "INV-2026-001" in result["invoice_ingest"]
    assert "refund request" in result["audio_ingest"]

    retrieved = result["retrieved_chunks"]
    assert len(retrieved) >= 2
    assert any("INV-2026-001" in chunk["content"] for chunk in retrieved)
    assert any("refund request" in chunk["content"] for chunk in retrieved)

    stages = [event["stage"] for event in result["trace"]]
    assert "perception.sync_rag" in stages
    assert "rag.ask" in stages
