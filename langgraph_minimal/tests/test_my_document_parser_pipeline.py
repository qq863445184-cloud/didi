from app.my_memory_system import DocumentParserPipeline, MyRAGTool


class FakeMarkItDown:
    def __init__(self) -> None:
        self.calls = []

    def convert(self, file_path):
        self.calls.append(file_path)

        class Result:
            text_content = "# Converted\n\n来自 MarkItDown 的内容"

        return Result()


class FakeEmbedder:
    def encode(self, texts):
        return [[1.0, 0.0] for _ in texts]


class FakeVectorStore:
    def __init__(self) -> None:
        self.rows = []

    def add_vectors(self, vectors, metadata, ids=None):
        row_ids = ids or [str(index) for index in range(len(vectors))]
        for vector, meta, row_id in zip(vectors, metadata, row_ids):
            self.rows.append({"id": row_id, "vector": vector, "metadata": meta})
        return True

    def search_similar(self, query_vector, limit=5, score_threshold=None, where=None):
        return []


class FakeLLM:
    def invoke(self, messages, **kwargs):
        return "answer"


def test_document_parser_pipeline_reads_plain_text_and_markdown(tmp_path):
    parser = DocumentParserPipeline()
    md_path = tmp_path / "note.md"
    md_path.write_text("# 标题\n\nRAG 文档内容", encoding="utf-8")

    parsed = parser.parse(md_path)

    assert parsed.text == "# 标题\n\nRAG 文档内容"
    assert parsed.parser == "plain_text"
    assert parsed.modality == "text"


def test_document_parser_pipeline_uses_injected_markitdown_for_office_files(tmp_path):
    markitdown = FakeMarkItDown()
    parser = DocumentParserPipeline(markitdown=markitdown)
    doc_path = tmp_path / "chapter8.docx"
    doc_path.write_bytes(b"fake office bytes")

    parsed = parser.parse(doc_path)

    assert parsed.text == "# Converted\n\n来自 MarkItDown 的内容"
    assert parsed.parser == "markitdown"
    assert parsed.modality == "document"
    assert markitdown.calls == [str(doc_path)]


def test_document_parser_pipeline_uses_ocr_for_images(tmp_path):
    parser = DocumentParserPipeline(image_ocr=lambda path: f"OCR:{path.name}")
    image_path = tmp_path / "diagram.png"
    image_path.write_bytes(b"fake image bytes")

    parsed = parser.parse(image_path)

    assert parsed.text == "OCR:diagram.png"
    assert parsed.parser == "ocr"
    assert parsed.modality == "image"


def test_document_parser_pipeline_uses_asr_for_audio(tmp_path):
    parser = DocumentParserPipeline(audio_asr=lambda path: f"ASR:{path.name}")
    audio_path = tmp_path / "talk.mp3"
    audio_path.write_bytes(b"fake audio bytes")

    parsed = parser.parse(audio_path)

    assert parsed.text == "ASR:talk.mp3"
    assert parsed.parser == "asr"
    assert parsed.modality == "audio"


def test_my_rag_tool_uses_parser_pipeline_for_add_document(tmp_path):
    parser = DocumentParserPipeline(image_ocr=lambda path: "图片中写着 RAG 流程")
    tool = MyRAGTool(
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
        llm=FakeLLM(),
        parser_pipeline=parser,
        collection_name="parser_pipeline_test",
    )
    image_path = tmp_path / "rag.png"
    image_path.write_bytes(b"fake image bytes")

    result = tool.run(
        {
            "action": "add_document",
            "file_path": str(image_path),
            "namespace": "chapter8",
        }
    )

    assert "解析器: ocr" in result
    assert tool.vector_store.rows[0]["metadata"]["content"] == "图片中写着 RAG 流程"
