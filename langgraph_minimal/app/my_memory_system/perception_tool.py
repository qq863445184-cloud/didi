from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Any, Callable

from hello_agents.tools.base import Tool, ToolParameter

from .manager import MyMemoryManager
from .stores import PerceptualMemoryStore, WorkingMemoryStore


class MyPerceptionTool(Tool):
    """Convert local files into perceptual memories.

    这个工具负责“感知解析入口”：识别文件模态、尽量提取可检索文本，
    然后把结果写入 MyMemoryManager 的 perceptual memory。
    """

    TEXT_SUFFIXES = {
        ".txt",
        ".md",
        ".py",
        ".js",
        ".ts",
        ".tsx",
        ".json",
        ".yaml",
        ".yml",
        ".csv",
        ".log",
    }
    IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
    AUDIO_SUFFIXES = {".mp3", ".wav", ".m4a", ".flac", ".ogg"}
    VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm"}

    def __init__(
        self,
        *,
        manager: MyMemoryManager | None = None,
        image_ocr: Callable[[Path], str] | None = None,
        audio_asr: Callable[[Path], str] | None = None,
        encoders: dict[str, Callable[[Path], list[float]]] | None = None,
        rag_tool: Any | None = None,
        rag_namespace: str = "perceptual",
    ) -> None:
        super().__init__(
            name="perception",
            description="多模态感知工具，支持将本地文件解析并写入感知记忆",
        )
        self.manager = manager or MyMemoryManager(
            stores={
                "working": WorkingMemoryStore(),
                "perceptual": PerceptualMemoryStore(),
            }
        )
        self.image_ocr = image_ocr or self._build_default_image_ocr()
        self.audio_asr = audio_asr
        self.encoders = encoders or {}
        self.rag_tool = rag_tool
        self.rag_namespace = rag_namespace
        self.trace_events: list[dict[str, Any]] = []
        self._call_trace: list[dict[str, Any]] = []

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="action",
                type="string",
                description="操作类型：ingest_file、ingest_directory、search、search_file、summary",
                required=True,
            ),
            ToolParameter(
                name="directory_path",
                type="string",
                description="要批量写入感知记忆的目录路径，action=ingest_directory 时使用",
                required=False,
            ),
            ToolParameter(
                name="file_path",
                type="string",
                description="要写入感知记忆的本地文件路径，action=ingest_file 时使用",
                required=False,
            ),
            ToolParameter(
                name="modality",
                type="string",
                description="可选模态覆盖：text、image、audio、video、structured、binary",
                required=False,
            ),
            ToolParameter(
                name="description",
                type="string",
                description="文件的人类可读描述，可帮助后续检索",
                required=False,
            ),
            ToolParameter(
                name="query",
                type="string",
                description="检索感知记忆的查询文本，action=search 时使用",
                required=False,
            ),
            ToolParameter(
                name="importance",
                type="number",
                description="记忆重要性，0 到 1",
                required=False,
                default=0.5,
            ),
            ToolParameter(
                name="limit",
                type="integer",
                description="返回结果数量",
                required=False,
                default=5,
            ),
        ]

    def validate_parameters(self, parameters: dict[str, Any]) -> bool:
        action = parameters.get("action")
        if action not in {"ingest_file", "ingest_directory", "search", "search_file", "summary"}:
            return False
        if action in {"ingest_file", "search_file"}:
            return bool(str(parameters.get("file_path", "")).strip())
        if action == "ingest_directory":
            return bool(str(parameters.get("directory_path", "")).strip())
        if action == "search":
            return bool(str(parameters.get("query", "")).strip())
        return True

    def run(self, parameters: dict[str, Any]) -> str:
        self._call_trace = []
        manager_trace_start = len(self.manager.trace_events)
        if not self.validate_parameters(parameters):
            return "Error: perception tool parameters are incomplete or action is unsupported"

        action = parameters["action"]
        try:
            if action == "ingest_file":
                result = self._ingest_file(parameters)
            elif action == "ingest_directory":
                result = self._ingest_directory(parameters)
            elif action == "search":
                result = self._search(parameters)
            elif action == "search_file":
                result = self._search_file(parameters)
            else:
                result = self._summary(parameters)
            self.trace_events.extend(self._call_trace)
            self.trace_events.extend(self.manager.trace_events[manager_trace_start:])
            return result
        except Exception as exc:
            self.trace_events.extend(self._call_trace)
            self.trace_events.extend(self.manager.trace_events[manager_trace_start:])
            return f"Error: {exc}"

    def _ingest_file(self, parameters: dict[str, Any]) -> str:
        file_path = Path(str(parameters["file_path"])).expanduser()
        if not file_path.exists():
            return f"Error: file does not exist: {file_path}"
        if not file_path.is_file():
            return f"Error: path is not a file: {file_path}"

        modality = str(parameters.get("modality") or self._detect_modality(file_path))
        self._call_trace.append(
            {
                "stage": "perception.detect_modality",
                "file_path": str(file_path),
                "modality": modality,
            }
        )

        extraction = self._extract(file_path, modality)
        embedding_meta = self._encode_file(file_path, modality)
        self._call_trace.append(
            {
                "stage": "perception.extract",
                "file_path": str(file_path),
                "modality": modality,
                "extractor": extraction["extractor"],
                "text_length": len(str(extraction.get("extracted_text", ""))),
            }
        )
        if embedding_meta:
            self._call_trace.append(
                {
                    "stage": "perception.encode",
                    "file_path": str(file_path),
                    "modality": modality,
                    "embedding_dim": embedding_meta["embedding_dim"],
                    "embedding_model": embedding_meta["embedding_model"],
                }
            )

        description = str(parameters.get("description") or "").strip()
        content = self._build_content(file_path, modality, description, extraction)
        record = self.manager.add(
            content=content,
            memory_type="perceptual",
            importance=float(parameters.get("importance", 0.5)),
            metadata={
                "modality": modality,
                "file_path": str(file_path),
                "mime_type": mimetypes.guess_type(str(file_path))[0] or "",
                "file_size": file_path.stat().st_size,
                "description": description,
                **extraction,
                **embedding_meta,
            },
        )
        rag_document_id = self._sync_extracted_text_to_rag(
            file_path=file_path,
            modality=modality,
            extraction=extraction,
            description=description,
        )
        semantic_memory_id = self._sync_extracted_text_to_semantic(
            file_path=file_path,
            modality=modality,
            extraction=extraction,
            description=description,
            importance=float(parameters.get("importance", 0.5)),
        )
        episodic_memory_id = self._sync_ingest_event_to_episodic(
            file_path=file_path,
            modality=modality,
            extraction=extraction,
            description=description,
            importance=float(parameters.get("importance", 0.5)),
            perceptual_memory_id=record.memory_id,
            rag_document_id=rag_document_id,
            semantic_memory_id=semantic_memory_id,
        )

        lines = [
            "Perceptual memory saved:",
            f"- id: {record.memory_id}",
            f"- modality: {modality}",
            f"- file_path: {file_path}",
            f"- extractor: {extraction['extractor']}",
        ]
        if extraction.get("extracted_text"):
            preview = str(extraction["extracted_text"]).replace("\n", " ")[:120]
            lines.append(f"- extracted_text: {preview}")
        if rag_document_id:
            lines.append(f"- rag_document_id: {rag_document_id}")
        if semantic_memory_id:
            lines.append(f"- semantic_memory_id: {semantic_memory_id}")
        if episodic_memory_id:
            lines.append(f"- episodic_memory_id: {episodic_memory_id}")
        if embedding_meta:
            lines.append(f"- embedding_dim: {embedding_meta['embedding_dim']}")
        return "\n".join(lines)

    def _ingest_directory(self, parameters: dict[str, Any]) -> str:
        directory_path = Path(str(parameters["directory_path"])).expanduser()
        if not directory_path.exists():
            return f"Error: directory does not exist: {directory_path}"
        if not directory_path.is_dir():
            return f"Error: path is not a directory: {directory_path}"

        files = sorted(path for path in directory_path.iterdir() if path.is_file())
        skipped = len([path for path in directory_path.iterdir() if not path.is_file()])
        ingested = 0
        failed: list[str] = []

        for file_path in files:
            # 批量入口复用单文件入口，保证模态检测、OCR/ASR、embedding 和 manager.add
            # 的行为完全一致；这里只负责目录级编排和汇总。
            result = self._ingest_file(
                {
                    **parameters,
                    "action": "ingest_file",
                    "file_path": str(file_path),
                }
            )
            if result.startswith("Error:"):
                failed.append(f"{file_path.name}: {result}")
            else:
                ingested += 1

        self._call_trace.append(
            {
                "stage": "perception.ingest_directory",
                "directory_path": str(directory_path),
                "files": len(files),
                "ingested": ingested,
                "skipped": skipped,
                "failed": len(failed),
            }
        )

        lines = [
            "Directory perceptual ingest finished:",
            f"- directory: {directory_path}",
            f"- files: {len(files)}",
            f"- ingested: {ingested}",
            f"- skipped: {skipped}",
            f"- failed: {len(failed)}",
        ]
        lines.extend(f"- {item}" for item in failed)
        return "\n".join(lines)

    def _search(self, parameters: dict[str, Any]) -> str:
        results = self.manager.search(
            query=str(parameters["query"]),
            memory_type="perceptual",
            limit=int(parameters.get("limit", 5)),
        )
        if not results:
            return "No perceptual memories found."

        lines = [f"Found {len(results)} perceptual memories:"]
        for index, item in enumerate(results, 1):
            meta = item.record.metadata
            lines.append(
                f"{index}. score={item.score:.2f} modality={meta.get('modality', '')} "
                f"file={meta.get('file_path', '')} content={item.record.content}"
            )
        return "\n".join(lines)

    def _search_file(self, parameters: dict[str, Any]) -> str:
        file_path = Path(str(parameters["file_path"])).expanduser()
        if not file_path.exists():
            return f"Error: file does not exist: {file_path}"

        modality = str(parameters.get("modality") or self._detect_modality(file_path))
        embedding_meta = self._encode_file(file_path, modality)
        if not embedding_meta.get("embedding"):
            return f"Error: no encoder is configured for modality={modality}"

        self._call_trace.append(
            {
                "stage": "perception.encode_query",
                "file_path": str(file_path),
                "modality": modality,
                "embedding_dim": embedding_meta["embedding_dim"],
            }
        )

        store = self.manager.stores.get("perceptual")
        searcher = getattr(store, "search_by_embedding", None)
        if searcher is None:
            return "Error: perceptual store does not support embedding search"

        results = searcher(
            embedding_meta["embedding"],
            modality=modality,
            limit=int(parameters.get("limit", 5)),
        )
        self._call_trace.append(
            {
                "stage": "perception.search_embedding",
                "file_path": str(file_path),
                "modality": modality,
                "hits": len(results),
            }
        )
        if not results:
            return "No perceptual memories found by file embedding."

        lines = [f"Found {len(results)} perceptual memories by file embedding:"]
        for index, item in enumerate(results, 1):
            meta = item.record.metadata
            lines.append(
                f"{index}. score={item.score:.2f} modality={meta.get('modality', '')} "
                f"file={meta.get('file_path', '')} content={item.record.content}"
            )
        return "\n".join(lines)

    def _summary(self, parameters: dict[str, Any]) -> str:
        records = self.manager.summary(
            memory_type="perceptual",
            limit=int(parameters.get("limit", 5)),
        )
        lines = [f"Perceptual memory count: {len(records)}"]
        for index, record in enumerate(records, 1):
            lines.append(
                f"{index}. modality={record.metadata.get('modality', '')} "
                f"file={record.metadata.get('file_path', '')} content={record.content}"
            )
        return "\n".join(lines)

    def _detect_modality(self, file_path: Path) -> str:
        suffix = file_path.suffix.lower()
        mime_type = mimetypes.guess_type(str(file_path))[0] or ""
        if suffix in self.TEXT_SUFFIXES or mime_type.startswith("text/"):
            return "text"
        if suffix in self.IMAGE_SUFFIXES or mime_type.startswith("image/"):
            return "image"
        if suffix in self.AUDIO_SUFFIXES or mime_type.startswith("audio/"):
            return "audio"
        if suffix in self.VIDEO_SUFFIXES or mime_type.startswith("video/"):
            return "video"
        if suffix in {".json", ".xml"}:
            return "structured"
        return "binary"

    def _extract(self, file_path: Path, modality: str) -> dict[str, Any]:
        if modality in {"text", "structured"}:
            return self._extract_text_file(file_path)
        if modality == "image":
            return self._extract_image_metadata(file_path)
        if modality == "audio":
            return self._extract_audio_metadata(file_path)
        if modality == "video":
            return self._extract_video_metadata(file_path)
        return {
            "extractor": "binary_metadata",
            "extracted_text": "",
            "extraction_note": "Binary file stored with metadata only.",
        }

    def _extract_text_file(self, file_path: Path) -> dict[str, Any]:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        return {
            "extractor": "text_reader",
            "extracted_text": text[:8000],
            "line_count": text.count("\n") + (1 if text else 0),
        }

    def _extract_image_metadata(self, file_path: Path) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "extractor": "image_metadata",
            "extracted_text": "",
            "extraction_note": "Image OCR/captioning is not enabled; stored metadata only.",
        }
        if self.image_ocr is not None:
            try:
                ocr_text = (self.image_ocr(file_path) or "").strip()
            except Exception as exc:
                metadata["ocr_error"] = str(exc)
            else:
                if ocr_text:
                    metadata["extractor"] = "image_ocr"
                    metadata["extracted_text"] = ocr_text[:8000]
                    metadata["extraction_note"] = "Image text extracted by OCR."
        try:
            from PIL import Image

            with Image.open(file_path) as image:
                metadata["image_width"] = image.width
                metadata["image_height"] = image.height
                metadata["image_format"] = image.format or ""
        except Exception as exc:
            metadata["image_error"] = str(exc)
        return metadata

    def _build_default_image_ocr(self) -> Callable[[Path], str] | None:
        """Use pytesseract when available, otherwise keep OCR optional.

        OCR 依赖通常包括 Python 包和本机 Tesseract 可执行文件。这里不强制安装，
        避免感知记忆因为缺少 OCR 环境而整体不可用。
        """

        try:
            import pytesseract
            from PIL import Image
        except Exception:
            return None

        def _ocr(file_path: Path) -> str:
            with Image.open(file_path) as image:
                return str(pytesseract.image_to_string(image))

        return _ocr

    def _extract_audio_metadata(self, file_path: Path) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "extractor": "audio_metadata",
            "extracted_text": "",
            "extraction_note": "Audio ASR is not enabled; stored metadata only.",
        }
        if self.audio_asr is not None:
            try:
                asr_text = (self.audio_asr(file_path) or "").strip()
            except Exception as exc:
                metadata["asr_error"] = str(exc)
            else:
                if asr_text:
                    metadata["extractor"] = "audio_asr"
                    metadata["extracted_text"] = asr_text[:8000]
                    metadata["extraction_note"] = "Audio text extracted by ASR."
        try:
            import wave

            with wave.open(str(file_path), "rb") as audio:
                metadata["audio_channels"] = audio.getnchannels()
                metadata["audio_sample_rate"] = audio.getframerate()
                metadata["audio_frames"] = audio.getnframes()
        except Exception as exc:
            metadata["audio_error"] = str(exc)
        return metadata

    def _extract_video_metadata(self, file_path: Path) -> dict[str, Any]:
        return {
            "extractor": "video_metadata",
            "extracted_text": "",
            "extraction_note": "Video parsing is not enabled; stored metadata only.",
        }

    def _encode_file(self, file_path: Path, modality: str) -> dict[str, Any]:
        encoder = self.encoders.get(modality)
        if encoder is None:
            return {}
        vector = encoder(file_path)
        normalized = [float(item) for item in vector]
        return {
            "embedding": normalized,
            "embedding_dim": len(normalized),
            "embedding_model": f"injected:{modality}",
        }

    def _sync_extracted_text_to_rag(
        self,
        *,
        file_path: Path,
        modality: str,
        extraction: dict[str, Any],
        description: str,
    ) -> str | None:
        """Write OCR/ASR/text extraction into RAG when a RAG tool is configured.

        感知记忆保存“看见/听见过什么”，RAG 保存“可被知识问答检索的文本”。
        只有 extracted_text 非空时才同步，避免把纯文件元数据塞进知识库。
        """

        if self.rag_tool is None:
            return None
        extracted_text = str(extraction.get("extracted_text") or "").strip()
        if not extracted_text:
            self._call_trace.append(
                {
                    "stage": "perception.skip_rag_sync",
                    "file_path": str(file_path),
                    "modality": modality,
                    "reason": "empty_extracted_text",
                }
            )
            return None

        document_id = f"perceptual:{modality}:{file_path.name}"
        rag_text_parts = [
            f"# Perceptual source: {file_path.name}",
            f"- modality: {modality}",
            f"- extractor: {extraction.get('extractor', '')}",
        ]
        if description:
            rag_text_parts.append(f"- description: {description}")
        rag_text_parts.extend(["", extracted_text])
        result = self.rag_tool.run(
            {
                "action": "add_text",
                "text": "\n".join(rag_text_parts),
                "document_id": document_id,
                "namespace": self.rag_namespace,
            }
        )
        self._call_trace.append(
            {
                "stage": "perception.sync_rag",
                "file_path": str(file_path),
                "modality": modality,
                "document_id": document_id,
                "namespace": self.rag_namespace,
                "result": result,
            }
        )
        return document_id

    def _sync_extracted_text_to_semantic(
        self,
        *,
        file_path: Path,
        modality: str,
        extraction: dict[str, Any],
        description: str,
        importance: float,
    ) -> str | None:
        if "semantic" not in self.manager.stores:
            return None
        extracted_text = str(extraction.get("extracted_text") or "").strip()
        if not extracted_text:
            self._call_trace.append(
                {
                    "stage": "perception.skip_semantic_sync",
                    "file_path": str(file_path),
                    "modality": modality,
                    "reason": "empty_extracted_text",
                }
            )
            return None

        content_parts = [
            f"Semantic knowledge extracted from {modality} file: {file_path.name}.",
        ]
        if description:
            content_parts.append(description)
        content_parts.append(extracted_text)
        record = self.manager.add(
            content=" ".join(content_parts),
            memory_type="semantic",
            importance=importance,
            metadata={
                "source": "perception",
                "modality": modality,
                "file_path": str(file_path),
                "extractor": extraction.get("extractor", ""),
                "description": description,
            },
        )
        self._call_trace.append(
            {
                "stage": "perception.sync_semantic",
                "file_path": str(file_path),
                "modality": modality,
                "memory_id": record.memory_id,
            }
        )
        return record.memory_id

    def _sync_ingest_event_to_episodic(
        self,
        *,
        file_path: Path,
        modality: str,
        extraction: dict[str, Any],
        description: str,
        importance: float,
        perceptual_memory_id: str,
        rag_document_id: str | None,
        semantic_memory_id: str | None,
    ) -> str | None:
        if "episodic" not in self.manager.stores:
            return None

        content_parts = [
            f"Perception ingest event for {modality} file: {file_path.name}.",
        ]
        if description:
            content_parts.append(description)
        if extraction.get("extracted_text"):
            content_parts.append(str(extraction["extracted_text"])[:500])
        record = self.manager.add(
            content=" ".join(content_parts),
            memory_type="episodic",
            importance=importance,
            metadata={
                "event_type": "perception_ingest",
                "source": "perception",
                "modality": modality,
                "file_path": str(file_path),
                "extractor": extraction.get("extractor", ""),
                "description": description,
                "perceptual_memory_id": perceptual_memory_id,
                "rag_document_id": rag_document_id,
                "semantic_memory_id": semantic_memory_id,
            },
        )
        self._call_trace.append(
            {
                "stage": "perception.sync_episodic",
                "file_path": str(file_path),
                "modality": modality,
                "memory_id": record.memory_id,
            }
        )
        return record.memory_id

    def _build_content(
        self,
        file_path: Path,
        modality: str,
        description: str,
        extraction: dict[str, Any],
    ) -> str:
        parts = [
            f"Perceptual observation from {modality} file: {file_path.name}.",
        ]
        if description:
            parts.append(description)
        if extraction.get("extracted_text"):
            parts.append(str(extraction["extracted_text"])[:1000])
        else:
            parts.append(str(extraction.get("extraction_note", "")))
        return " ".join(part for part in parts if part).strip()
