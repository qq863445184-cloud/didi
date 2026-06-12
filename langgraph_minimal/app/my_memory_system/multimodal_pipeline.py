from __future__ import annotations

import json
import os
import subprocess
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


class ExternalRuntimeProcessor:
    """Call a separate Python runtime for heavyweight multimodal inference.

    FunASR/PaddleOCR 经常对 Python 版本和二进制依赖比较敏感。这个适配器把
    主 Agent 进程和真实 OCR/ASR 推理进程解耦：主进程只通过 JSON 文本交换结果，
    重模型加载留在 `.venv-asr` 这类专门运行时中完成。
    """

    def __init__(
        self,
        *,
        action: str,
        python_path: str | Path,
        worker_path: str | Path,
        model_dir: str | Path,
        timeout_seconds: float = 180.0,
    ) -> None:
        self.action = action
        self.python_path = Path(python_path)
        self.worker_path = Path(worker_path)
        self.model_dir = Path(model_dir)
        self.timeout_seconds = timeout_seconds

    def __call__(self, file_path: str | Path) -> str:
        path = Path(file_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Multimodal input file does not exist: {path}")
        if not self.python_path.exists():
            raise FileNotFoundError(f"Multimodal Python runtime does not exist: {self.python_path}")
        if not self.worker_path.exists():
            raise FileNotFoundError(f"Multimodal worker script does not exist: {self.worker_path}")

        completed = subprocess.run(
            [
                str(self.python_path),
                str(self.worker_path),
                self.action,
                "--file",
                str(path),
                "--model-dir",
                str(self.model_dir),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=self.timeout_seconds,
            env=os.environ.copy(),
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "External multimodal runtime failed: "
                f"exit={completed.returncode} stderr={completed.stderr.strip()}"
            )
        return self._extract_text(completed.stdout)

    def _extract_text(self, output: str) -> str:
        stripped = output.strip()
        if not stripped:
            return ""
        last_line = stripped.splitlines()[-1]
        try:
            payload = json.loads(last_line)
        except json.JSONDecodeError:
            return stripped
        if isinstance(payload, dict):
            return str(payload.get("text") or payload.get("result") or "").strip()
        return str(payload).strip()


class ExternalRuntimeASR(ExternalRuntimeProcessor):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(action="asr", **kwargs)


class ExternalRuntimeOCR(ExternalRuntimeProcessor):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(action="ocr", **kwargs)


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
    multimodal_python: str | Path | None = None,
    multimodal_worker: str | Path | None = None,
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
        if multimodal_python or multimodal_worker:
            resolved_image_ocr = ExternalRuntimeOCR(
                python_path=multimodal_python or Path(".venv-asr") / "Scripts" / "python.exe",
                worker_path=multimodal_worker or Path("scripts") / "chapter8_multimodal_worker.py",
                model_dir=model_root / "PaddlePaddle" / "PaddleOCR-VL-1.6",
            )
        else:
            resolved_image_ocr = PaddleOCRVLOCR(
                model_dir=model_root / "PaddlePaddle" / "PaddleOCR-VL-1.6"
            )

    resolved_audio_asr = audio_asr
    if resolved_audio_asr is None and enable_asr:
        if multimodal_python or multimodal_worker:
            resolved_audio_asr = ExternalRuntimeASR(
                python_path=multimodal_python or Path(".venv-asr") / "Scripts" / "python.exe",
                worker_path=multimodal_worker or Path("scripts") / "chapter8_multimodal_worker.py",
                model_dir=model_root / "iic" / "SenseVoiceSmall",
            )
        else:
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
