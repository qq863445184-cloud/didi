from scripts.chapter8_neo4j_graph_smoke import run_smoke


def test_chapter8_neo4j_graph_smoke_runs_with_fake_graph_store():
    result = run_smoke(smoke_id="unit")

    assert result["semantic"]["graph_hit_count"] >= 1
    assert result["semantic"]["first_hit_source"] == "semantic_graph"
    assert result["episodic"]["graph_hit_count"] >= 1
    assert result["episodic"]["first_hit_source"] == "episodic"
    assert result["cleanup"]["attempted"] is False
