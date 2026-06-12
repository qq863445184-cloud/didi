# Chapter 8 Memory And Retrieval Implementation

This project implements a Chapter 8 style memory and retrieval stack on top of
hello-agents without changing the upstream package source.

## Architecture

The custom memory layer is organized around working / semantic / episodic / perceptual
memory:

- working memory keeps short-lived task context and supports JSON persistence.
- semantic memory stores stable knowledge, writes vectors to Qdrant, and can index
  extracted entities into Neo4j.
- episodic memory stores concrete events in Neo4j as user-to-episode graph records.
- perceptual memory stores files and multimodal observations, including extracted
  OCR/ASR text, file metadata, and optional CLIP/CLAP embeddings.

All stores use the same `MemoryRecord` and `MemorySearchResult` shape. Search scores
combine base similarity, recency, importance, and access reinforcement, so frequently
retrieved memories become harder to forget.

## Tools

`MyMemoryTool` is the agent-facing memory tool. It exposes add, search, summary,
stats, update, remove, clear_all, forget, and consolidate operations.

`MyRAGTool` is the retrieval tool. It supports add_text, add_document,
delete_document, search, ask, and stats. Documents are parsed into Markdown-like text,
split into heading-aware chunks, stored in SQLite metadata, and indexed into a vector
store for retrieval. The ask action follows the normal RAG flow: retrieve chunks,
build context, call the LLM, and keep trace events for inspection.

`MyPerceptionTool` is the multimodal entry point. It detects text, image, audio,
video, structured, and binary files, stores perceptual records, and syncs extracted
text into semantic memory, episodic memory, and RAG when those backends are configured.

## Backends

Qdrant is used for vector similarity search. Neo4j is used for graph memory and
entity/event relationships. SQLite is used for RAG document and chunk metadata.
spaCy is used for entity extraction when semantic graph indexing is enabled.

Heavy OCR/ASR packages can live outside the main Python environment. The main `.venv`
can call `.venv-asr` through `chapter8_multimodal_worker.py`, which keeps
FunASR/PaddleOCR dependencies isolated from the agent runtime.

## Multimodal Runtime

The default lightweight demo uses injected OCR/ASR and deterministic embeddings.
For a real OCR/ASR runtime, configure or use the default external Python:

```powershell
.\.venv\Scripts\python.exe scripts\chapter8_backend_health.py --check-services --timeout 2
.\.venv-asr\Scripts\python.exe scripts\chapter8_multimodal_worker.py check
```

The worker supports:

```powershell
.\.venv-asr\Scripts\python.exe scripts\chapter8_multimodal_worker.py asr --file path\to\audio.wav --model-dir models\iic\SenseVoiceSmall
.\.venv-asr\Scripts\python.exe scripts\chapter8_multimodal_worker.py ocr --file path\to\image.png --model-dir models\PaddlePaddle\PaddleOCR-VL-1.6
```

For stable local smoke tests without loading large models, set
`CHAPTER8_FAKE_MULTIMODAL=1`.

## Smoke Commands

Run backend readiness:

```powershell
.\.venv\Scripts\python.exe scripts\chapter8_backend_health.py --check-services --timeout 2
```

Run deterministic RAG/document learning demos:

```powershell
.\.venv\Scripts\python.exe scripts\chapter8_document_learning_demo.py
.\.venv\Scripts\python.exe scripts\chapter8_multimodal_pipeline_demo.py
```

Run Neo4j graph memory smoke with a fake graph store:

```powershell
.\.venv\Scripts\python.exe scripts\chapter8_neo4j_graph_smoke.py --smoke-id local_demo
```

Run the real Neo4j Aura smoke and clean temporary nodes:

```powershell
.\.venv\Scripts\python.exe scripts\chapter8_neo4j_graph_smoke.py --real --smoke-id chapter8_real_smoke
```

The real smoke verifies both semantic graph recall and episodic graph recall, then
deletes nodes whose id, name, content, or user_id contains the smoke id.

## Test Gates

Useful Chapter 8 regression commands:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_chapter8_backend_health.py tests\test_my_multimodal_pipeline.py tests\test_my_perception_tool.py -q
.\.venv\Scripts\python.exe -m pytest tests\test_my_rag_document_store.py tests\test_my_rag_tool.py tests\test_chapter8_architecture_coverage.py -q
.\.venv\Scripts\python.exe -m pytest tests\test_chapter8_neo4j_graph_smoke.py tests\test_my_semantic_graph.py tests\test_my_memory_persistence.py -q
```
