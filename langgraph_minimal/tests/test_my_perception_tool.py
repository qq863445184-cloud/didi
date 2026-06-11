from pathlib import Path

from app.my_memory_system import MyMemoryManager, MyPerceptionTool, PerceptualMemoryStore


def test_perception_tool_ingests_text_file_into_perceptual_memory(tmp_path):
    sample = tmp_path / "note.md"
    sample.write_text("Agent memory can ingest Python code screenshots and notes.", encoding="utf-8")
    manager = MyMemoryManager(
        user_id="perception_user",
        stores={"perceptual": PerceptualMemoryStore()},
    )
    tool = MyPerceptionTool(manager=manager)

    result = tool.run(
        {
            "action": "ingest_file",
            "file_path": str(sample),
            "importance": 0.8,
            "description": "A markdown note about Agent memory.",
        }
    )
    search_result = tool.run(
        {
            "action": "search",
            "query": "Python code screenshots",
            "limit": 3,
        }
    )

    assert "Perceptual memory saved" in result
    assert "modality: text" in result
    assert "extracted_text" in result
    assert "Found 1 perceptual memories" in search_result
    assert "Python code screenshots" in search_result
    assert tool.trace_events[0]["stage"] == "perception.detect_modality"
    assert tool.trace_events[1]["stage"] == "perception.extract"
    assert tool.trace_events[2]["stage"] == "manager.add"


def test_perception_tool_records_image_metadata_without_ocr(tmp_path):
    image = tmp_path / "diagram.png"
    image.write_bytes(b"not-a-real-png-but-still-a-file")
    manager = MyMemoryManager(
        user_id="perception_user",
        stores={"perceptual": PerceptualMemoryStore()},
    )
    tool = MyPerceptionTool(manager=manager)

    result = tool.run(
        {
            "action": "ingest_file",
            "file_path": str(image),
            "description": "Architecture diagram uploaded by user.",
        }
    )
    search_result = tool.run({"action": "search", "query": "Architecture diagram"})

    assert "modality: image" in result
    assert "image_metadata" in result
    assert "Found 1 perceptual memories" in search_result
    assert "Architecture diagram" in search_result


def test_perception_tool_uses_injected_ocr_for_image_text(tmp_path):
    image = tmp_path / "screenshot.png"
    image.write_bytes(b"fake image bytes")
    manager = MyMemoryManager(
        user_id="perception_user",
        stores={"perceptual": PerceptualMemoryStore()},
    )
    tool = MyPerceptionTool(
        manager=manager,
        image_ocr=lambda path: "截图中包含函数 calculate_total 和订单金额字段",
    )

    result = tool.run({"action": "ingest_file", "file_path": str(image)})
    search_result = tool.run({"action": "search", "query": "calculate_total 订单金额"})

    assert "extractor: image_ocr" in result
    assert "calculate_total" in result
    assert "Found 1 perceptual memories" in search_result
    assert "订单金额" in search_result


def test_perception_tool_uses_injected_asr_for_audio_text(tmp_path):
    audio = tmp_path / "meeting.wav"
    audio.write_bytes(b"fake wav bytes")
    manager = MyMemoryManager(
        user_id="perception_user",
        stores={"perceptual": PerceptualMemoryStore()},
    )
    tool = MyPerceptionTool(
        manager=manager,
        audio_asr=lambda path: "会议中讨论了 Qdrant 向量检索和 Neo4j 图谱记忆",
    )

    result = tool.run({"action": "ingest_file", "file_path": str(audio)})
    search_result = tool.run({"action": "search", "query": "Qdrant Neo4j 图谱记忆"})

    assert "modality: audio" in result
    assert "extractor: audio_asr" in result
    assert "Qdrant" in result
    assert "Found 1 perceptual memories" in search_result


def test_perception_tool_attaches_modality_embedding_when_encoder_is_injected(tmp_path):
    image = tmp_path / "chart.png"
    image.write_bytes(b"fake chart bytes")
    manager = MyMemoryManager(
        user_id="perception_user",
        stores={"perceptual": PerceptualMemoryStore()},
    )
    tool = MyPerceptionTool(
        manager=manager,
        encoders={"image": lambda path: [0.1, 0.2, 0.3]},
    )

    result = tool.run({"action": "ingest_file", "file_path": str(image)})
    record = manager.stores["perceptual"].records[0]

    assert "embedding_dim: 3" in result
    assert record.metadata["embedding"] == [0.1, 0.2, 0.3]
    assert record.metadata["embedding_model"] == "injected:image"


def test_perception_tool_searches_similar_file_by_modality_embedding(tmp_path):
    chart_a = tmp_path / "chart_a.png"
    chart_b = tmp_path / "chart_b.png"
    query_chart = tmp_path / "query_chart.png"
    chart_a.write_bytes(b"chart a")
    chart_b.write_bytes(b"chart b")
    query_chart.write_bytes(b"query chart")
    manager = MyMemoryManager(
        user_id="perception_user",
        stores={"perceptual": PerceptualMemoryStore()},
    )

    vectors = {
        chart_a.name: [1.0, 0.0],
        chart_b.name: [0.0, 1.0],
        query_chart.name: [0.9, 0.1],
    }
    tool = MyPerceptionTool(
        manager=manager,
        encoders={"image": lambda path: vectors[Path(path).name]},
    )

    tool.run(
        {
            "action": "ingest_file",
            "file_path": str(chart_a),
            "description": "支付流程架构图",
        }
    )
    tool.run(
        {
            "action": "ingest_file",
            "file_path": str(chart_b),
            "description": "音频处理流程图",
        }
    )
    result = tool.run(
        {
            "action": "search_file",
            "file_path": str(query_chart),
            "limit": 1,
        }
    )

    assert "Found 1 perceptual memories by file embedding" in result
    assert "支付流程架构图" in result
    assert "音频处理流程图" not in result
    assert tool.trace_events[-2]["stage"] == "perception.encode_query"
    assert tool.trace_events[-1]["stage"] == "perception.search_embedding"


def test_perception_tool_returns_clear_error_for_missing_file(tmp_path):
    tool = MyPerceptionTool()

    result = tool.run({"action": "ingest_file", "file_path": str(tmp_path / "missing.png")})

    assert "Error: file does not exist" in result
