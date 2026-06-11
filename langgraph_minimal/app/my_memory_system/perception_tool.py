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
        self.trace_events: list[dict[str, Any]] = []
        self._call_trace: list[dict[str, Any]] = []

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="action",
                type="string",
                description="操作类型：ingest_file、search、summary",
                required=True,
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
        if action not in {"ingest_file", "search", "summary"}:
            return False
        if action == "ingest_file":
            return bool(str(parameters.get("file_path", "")).strip())
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
            elif action == "search":
                result = self._search(parameters)
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
        if embedding_meta:
            lines.append(f"- embedding_dim: {embedding_meta['embedding_dim']}")
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
