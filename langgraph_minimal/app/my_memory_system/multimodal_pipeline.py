from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .manager import MyMemoryManager
from .multimodal_encoders import ClapAudioEmbedder, ClipImageEmbedder
from .perception_tool import MyPerceptionTool
from .rag_tool import MyRAGTool
from .stores import PerceptualMemoryStore


class SenseVoiceASR:
    """Lazy SenseVoice ASR adapter for MyPerceptionTool.

    ASR 模型很重，所以这里不在 __init__ 加载。只有真正处理音频文件时，
    才 import funasr 并加载本地模型目录。
    """

    def __init__(
        self,
        *,
        model_dir: str | Path = "models/iic/SenseVoiceSmall",
        model_loader: Callable[[Path], Any] | None = None,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.model_loader = model_loader or self._load_model
        self._model: Any | None = None

    def __call__(self, file_path: str | Path) -> str:
        path = Path(file_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Audio file does not exist: {path}")
        model = self._ensure_model()
        result = model.generate(
            input=str(path),
            language="auto",
            use_itn=True,
            batch_size_s=60,
        )
        return self._extract_text(result)

    def _ensure_model(self) -> Any:
        if self._model is None:
            self._model = self.model_loader(self.model_dir)
        return self._model

    def _load_model(self, model_dir: Path) -> Any:
        from funasr import AutoModel

        return AutoModel(model=str(model_dir))

    def _extract_text(self, result: Any) -> str:
        if isinstance(result, str):
            return result.strip()
        if isinstance(result, dict):
            return str(result.get("text") or result.get("sentence_info") or result).strip()
        if isinstance(result, list):
            parts = [self._extract_text(item) for item in result]
            return "\n".join(part for part in parts if part).strip()
        return str(result).strip()


class PaddleOCRVLOCR:
    """Lazy PaddleOCR-VL adapter for image/document OCR."""

    def __init__(
        self,
        *,
        model_dir: str | Path = "models/PaddlePaddle/PaddleOCR-VL-1.6",
        pipeline_version: str = "v1.6",
        model_loader: Callable[[Path, str], Any] | None = None,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.pipeline_version = pipeline_version
        self.model_loader = model_loader or self._load_model
        self._model: Any | None = None

    def __call__(self, file_path: str | Path) -> str:
        path = Path(file_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Image file does not exist: {path}")
        model = self._ensure_model()
        result = model.predict(str(path))
        return self._extract_text(result)

    def _ensure_model(self) -> Any:
        if self._model is None:
            self._model = self.model_loader(self.model_dir, self.pipeline_version)
        return self._model

    def _load_model(self, model_dir: Path, pipeline_version: str) -> Any:
        from paddleocr import PaddleOCRVL

        # 显式传本地目录，避免 PaddleOCR 默认去用户目录重新下载模型。
        return PaddleOCRVL(
            pipeline_version=pipeline_version,
            vl_rec_model_dir=str(model_dir),
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
        )

    def _extract_text(self, result: Any) -> str:
        texts = list(self._iter_text_values(result))
        return "\n".join(text for text in texts if text).strip()

    def _iter_text_values(self, value: Any):
        if value is None:
            return
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                yield stripped
            return
        if isinstance(value, dict):
            for key in ("text", "rec_text", "content", "markdown_text"):
                text = value.get(key)
                if isinstance(text, str) and text.strip():
                    yield text.strip()
            for item in value.values():
                yield from self._iter_text_values(item)
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                yield from self._iter_text_values(item)
            return
        if hasattr(value, "json"):
            yield from self._iter_text_values(value.json)
            return
        if hasattr(value, "to_dict"):
            yield from self._iter_text_values(value.to_dict())


def build_multimodal_perception_tool(
    *,
    manager: MyMemoryManager | None = None,
    rag_tool: Any | None = None,
    rag_namespace: str = "multimodal",
    image_ocr: Callable[[Path], str] | None = None,
    audio_asr: Callable[[Path], str] | None = None,
    image_encoder: Callable[[Path], list[float]] | None = None,
    audio_encoder: Callable[[Path], list[float]] | None = None,
    enable_ocr: bool = False,
    enable_asr: bool = False,
    enable_image_embedding: bool = True,
    enable_audio_embedding: bool = True,
    enable_rag: bool = True,
    model_root: str | Path = "models",
    collection_name: str = "chapter8_multimodal_rag",
) -> MyPerceptionTool:
    """Build one ready-to-use multimodal perception pipeline.

    这个函数把“可插拔散件”收束成默认流水线：
    - perceptual memory 保存文件观察；
    - OCR/ASR 抽取文本；
    - CLIP/CLAP 生成模态向量；
    - RAGTool 同步可检索文本。
    """

    model_root = Path(model_root)
    manager = manager or MyMemoryManager(
        stores={"perceptual": PerceptualMemoryStore()},
    )
    if enable_rag and rag_tool is None:
        rag_tool = MyRAGTool(collection_name=collection_name)

    resolved_image_ocr = image_ocr
    if resolved_image_ocr is None and enable_ocr:
        resolved_image_ocr = PaddleOCRVLOCR(
            model_dir=model_root / "PaddlePaddle" / "PaddleOCR-VL-1.6"
        )

    resolved_audio_asr = audio_asr
    if resolved_audio_asr is None and enable_asr:
        resolved_audio_asr = SenseVoiceASR(model_dir=model_root / "iic" / "SenseVoiceSmall")

    encoders: dict[str, Callable[[Path], list[float]]] = {}
    if image_encoder is not None:
        encoders["image"] = image_encoder
    elif enable_image_embedding:
        encoders["image"] = ClipImageEmbedder()

    if audio_encoder is not None:
        encoders["audio"] = audio_encoder
    elif enable_audio_embedding:
        encoders["audio"] = ClapAudioEmbedder()

    return MyPerceptionTool(
        manager=manager,
        image_ocr=resolved_image_ocr,
        audio_asr=resolved_audio_asr,
        encoders=encoders,
        rag_tool=rag_tool,
        rag_namespace=rag_namespace,
    )
