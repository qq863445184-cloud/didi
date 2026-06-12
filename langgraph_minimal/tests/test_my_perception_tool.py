from pathlib import Path

from app.my_memory_system import MyMemoryManager, MyPerceptionTool, MyRAGTool, PerceptualMemoryStore


class FakeEmbedder:
    def encode(self, texts):
        vectors = []
        for text in texts:
            if "Qdrant" in text or "Neo4j" in text or "图谱记忆" in text:
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.0, 1.0])
        return vectors


class FakeVectorStore:
    def __init__(self) -> None:
        self.rows = []

    def add_vectors(self, vectors, metadata, ids=None):
        ids = ids or [str(index) for index in range(len(vectors))]
        for vector, meta, row_id in zip(vectors, metadata, ids):
            self.rows.append({"id": row_id, "vector": vector, "metadata": meta})
        return True

    def search_similar(self, query_vector, limit=5, score_threshold=None, where=None):
        hits = []
        for row in self.rows:
            if where and not all(row["metadata"].get(key) == value for key, value in where.items()):
                continue
            score = sum(a * b for a, b in zip(query_vector, row["vector"]))
            if score_threshold is not None and score < score_threshold:
                continue
            hits.append({"id": row["id"], "score": score, "metadata": row["metadata"]})
        hits.sort(key=lambda item: item["score"], reverse=True)
        return hits[:limit]


class FakeLLM:
    def invoke(self, messages, **kwargs):
        return "音频内容提到了 Qdrant 向量检索和 Neo4j 图谱记忆。"


def build_fake_rag_tool():
    return MyRAGTool(
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
        llm=FakeLLM(),
        collection_name="perception_rag_test",
    )


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


def test_perception_tool_syncs_audio_asr_text_to_rag_when_configured(tmp_path):
    audio = tmp_path / "meeting.mp3"
    audio.write_bytes(b"fake mp3 bytes")
    manager = MyMemoryManager(
        user_id="perception_user",
        stores={"perceptual": PerceptualMemoryStore()},
    )
    rag_tool = build_fake_rag_tool()
    tool = MyPerceptionTool(
        manager=manager,
        rag_tool=rag_tool,
        rag_namespace="multimodal",
        audio_asr=lambda path: "会议中讨论了 Qdrant 向量检索和 Neo4j 图谱记忆",
    )

    ingest_result = tool.run({"action": "ingest_file", "file_path": str(audio)})
    rag_result = rag_tool.run(
        {
            "action": "search",
            "query": "Qdrant Neo4j 图谱记忆",
            "namespace": "multimodal",
        }
    )

    assert "rag_document_id" in ingest_result
    assert "搜索结果" in rag_result
    assert "meeting.mp3" in rag_result
    assert any(event["stage"] == "perception.sync_rag" for event in tool.trace_events)


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


def test_perception_tool_ingests_directory_of_multimodal_files(tmp_path):
    note = tmp_path / "agent_note.md"
    image = tmp_path / "architecture.png"
    audio = tmp_path / "meeting.wav"
    nested_dir = tmp_path / "nested"
    nested_dir.mkdir()
    note.write_text("第八章讨论 Agent 记忆、RAG 检索和知识库问答。", encoding="utf-8")
    image.write_bytes(b"fake image")
    audio.write_bytes(b"fake wav")
    manager = MyMemoryManager(
        user_id="perception_user",
        stores={"perceptual": PerceptualMemoryStore()},
    )
    tool = MyPerceptionTool(
        manager=manager,
        image_ocr=lambda path: "架构图展示 Qdrant 和 Neo4j 的连接关系",
        audio_asr=lambda path: "会议提到感知记忆需要支持 OCR 和 ASR",
    )

    result = tool.run(
        {
            "action": "ingest_directory",
            "directory_path": str(tmp_path),
            "description": "第八章多模态资料目录",
            "importance": 0.75,
        }
    )
    search_result = tool.run({"action": "search", "query": "OCR ASR Qdrant Neo4j"})

    assert "Directory perceptual ingest finished" in result
    assert "- ingested: 3" in result
    assert "- skipped: 1" in result
    assert len(manager.stores["perceptual"].records) == 3
    assert "Found 2 perceptual memories" in search_result
    assert any(event["stage"] == "perception.ingest_directory" for event in tool.trace_events)


def test_perception_tool_returns_clear_error_for_missing_file(tmp_path):
    tool = MyPerceptionTool()

    result = tool.run({"action": "ingest_file", "file_path": str(tmp_path / "missing.png")})

    assert "Error: file does not exist" in result
