from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass
class ParsedDocument:
    """Normalized parser output for RAG ingestion."""

    text: str
    parser: str
    modality: str
    metadata: dict[str, Any] = field(default_factory=dict)


class DocumentParserPipeline:
    """Convert common document and media files into indexable text.

    第八章的 RAG 入口不应该散落在 RAGTool 里。这个 pipeline 统一负责：
    - 纯文本/Markdown 直接读取；
    - PDF/Office 等结构化文档走 MarkItDown；
    - 图片走 OCR；
    - 音频/视频走 ASR。

    OCR/ASR/MarkItDown 都可注入，单元测试不用加载真实大模型。
    """

    TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".json", ".csv", ".log", ".py"}
    DOCUMENT_SUFFIXES = {
        ".pdf",
        ".doc",
        ".docx",
        ".ppt",
        ".pptx",
        ".xls",
        ".xlsx",
        ".html",
        ".htm",
    }
    IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
    AUDIO_SUFFIXES = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}
    VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm"}

    def __init__(
        self,
        *,
        markitdown: Any | None = None,
        image_ocr: Callable[[Path], str] | None = None,
        audio_asr: Callable[[Path], str] | None = None,
    ) -> None:
        self.markitdown = markitdown
        self.image_ocr = image_ocr
        self.audio_asr = audio_asr

    def parse(self, file_path: str | Path) -> ParsedDocument:
        path = Path(file_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")
        if not path.is_file():
            raise ValueError(f"不是文件: {path}")

        suffix = path.suffix.lower()
        if suffix in self.TEXT_SUFFIXES:
            return ParsedDocument(
                text=path.read_text(encoding="utf-8"),
                parser="plain_text",
                modality="text",
                metadata={"source_path": str(path), "suffix": suffix},
            )
        if suffix in self.IMAGE_SUFFIXES:
            return self._parse_image(path, suffix)
        if suffix in self.AUDIO_SUFFIXES:
            return self._parse_audio(path, suffix)
        if suffix in self.VIDEO_SUFFIXES:
            return self._parse_audio(path, suffix, modality="video")
        return self._parse_with_markitdown(path, suffix)

    def _parse_image(self, path: Path, suffix: str) -> ParsedDocument:
        if self.image_ocr is None:
            raise RuntimeError(f"当前图片类型 {suffix} 需要配置 OCR 解析器")
        return ParsedDocument(
            text=str(self.image_ocr(path)),
            parser="ocr",
            modality="image",
            metadata={"source_path": str(path), "suffix": suffix},
        )

    def _parse_audio(self, path: Path, suffix: str, *, modality: str = "audio") -> ParsedDocument:
        if self.audio_asr is None:
            raise RuntimeError(f"当前{modality}类型 {suffix} 需要配置 ASR 解析器")
        return ParsedDocument(
            text=str(self.audio_asr(path)),
            parser="asr",
            modality=modality,
            metadata={"source_path": str(path), "suffix": suffix},
        )

    def _parse_with_markitdown(self, path: Path, suffix: str) -> ParsedDocument:
        converter = self.markitdown or self._build_markitdown()
        converted = converter.convert(str(path))
        text = getattr(converted, "text_content", None) or str(converted)
        return ParsedDocument(
            text=text,
            parser="markitdown",
            modality="document",
            metadata={"source_path": str(path), "suffix": suffix},
        )

    def _build_markitdown(self) -> Any:
        try:
            from markitdown import MarkItDown
        except Exception as exc:
            raise RuntimeError("当前文件类型需要安装 markitdown 后才能解析") from exc
        return MarkItDown()
