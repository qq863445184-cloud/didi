from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Sequence


def _l2_normalize(vector: Sequence[float]) -> list[float]:
    """Normalize a vector so dot product can be used as cosine similarity."""

    values = [float(item) for item in vector]
    norm = math.sqrt(sum(item * item for item in values))
    if norm == 0:
        raise ValueError("zero vector cannot be normalized")
    return [item / norm for item in values]


def _to_flat_vector(features: Any) -> list[float]:
    """Convert model output tensors or test doubles into a single vector."""

    # Some transformers versions return BaseModelOutputWithPooling instead of
    # a bare tensor from get_image_features/get_audio_features.
    for attribute in ("pooler_output", "image_embeds", "audio_embeds", "last_hidden_state"):
        value = getattr(features, attribute, None)
        if value is not None:
            features = value
            break

    if hasattr(features, "detach"):
        features = features.detach()
    if hasattr(features, "cpu"):
        features = features.cpu()
    if hasattr(features, "tolist"):
        features = features.tolist()

    if isinstance(features, tuple):
        features = list(features)
    if not isinstance(features, list):
        raise TypeError(f"Unsupported feature output type: {type(features)!r}")
    if features and isinstance(features[0], list):
        features = features[0]
    return [float(item) for item in features]


class ClipImageEmbedder:
    """CLIP image encoder for perceptual memory.

    模型和 processor 都是懒加载：创建工具时不加载大模型，只有真正写入或检索
    图片文件时才加载一次。测试也可以注入 fake loader，避免单测依赖真实模型。
    """

    def __init__(
        self,
        *,
        model_name: str = "openai/clip-vit-base-patch32",
        local_files_only: bool = True,
        model_loader: Callable[[str, bool], Any] | None = None,
        processor_loader: Callable[[str, bool], Any] | None = None,
        image_loader: Callable[[Path], Any] | None = None,
    ) -> None:
        self.model_name = model_name
        self.local_files_only = local_files_only
        self.model_loader = model_loader or self._load_model
        self.processor_loader = processor_loader or self._load_processor
        self.image_loader = image_loader or self._load_image
        self._model: Any | None = None
        self._processor: Any | None = None

    def __call__(self, file_path: str | Path) -> list[float]:
        path = Path(file_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Image file does not exist: {path}")

        model, processor = self._ensure_loaded()
        image = self.image_loader(path)
        inputs = processor(images=image, return_tensors="pt")
        inputs = self._move_inputs_to_model_device(inputs, model)
        features = self._extract_features(model, inputs)
        return _l2_normalize(_to_flat_vector(features))

    def _ensure_loaded(self) -> tuple[Any, Any]:
        if self._model is None:
            self._model = self.model_loader(self.model_name, self.local_files_only)
            if hasattr(self._model, "eval"):
                self._model.eval()
        if self._processor is None:
            self._processor = self.processor_loader(self.model_name, self.local_files_only)
        return self._model, self._processor

    def _extract_features(self, model: Any, inputs: dict[str, Any]) -> Any:
        # Import torch only for real inference; fake unit-test models do not need it.
        try:
            import torch
        except Exception:
            return model.get_image_features(**inputs)
        with torch.no_grad():
            return model.get_image_features(**inputs)

    def _move_inputs_to_model_device(self, inputs: Any, model: Any) -> Any:
        device = getattr(model, "device", None)
        if device is not None and hasattr(inputs, "to"):
            return inputs.to(device)
        return inputs

    def _load_model(self, model_name: str, local_files_only: bool) -> Any:
        from transformers import CLIPModel

        return CLIPModel.from_pretrained(model_name, local_files_only=local_files_only)

    def _load_processor(self, model_name: str, local_files_only: bool) -> Any:
        from transformers import CLIPProcessor

        return CLIPProcessor.from_pretrained(model_name, local_files_only=local_files_only)

    def _load_image(self, path: Path) -> Any:
        from PIL import Image

        return Image.open(path).convert("RGB")


class ClapAudioEmbedder:
    """CLAP audio encoder for perceptual memory.

    默认使用 transformers 的 CLAP 模型，并通过 librosa 读取音频。输出同样做 L2
    归一化，以匹配 PerceptualMemoryStore 当前的点积相似度检索。
    """

    def __init__(
        self,
        *,
        model_name: str = "laion/clap-htsat-unfused",
        target_sampling_rate: int = 48000,
        local_files_only: bool = True,
        model_loader: Callable[[str, bool], Any] | None = None,
        processor_loader: Callable[[str, bool], Any] | None = None,
        audio_loader: Callable[[Path, int], tuple[Any, int]] | None = None,
    ) -> None:
        self.model_name = model_name
        self.target_sampling_rate = target_sampling_rate
        self.local_files_only = local_files_only
        self.model_loader = model_loader or self._load_model
        self.processor_loader = processor_loader or self._load_processor
        self.audio_loader = audio_loader or self._load_audio
        self._model: Any | None = None
        self._processor: Any | None = None

    def __call__(self, file_path: str | Path) -> list[float]:
        path = Path(file_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Audio file does not exist: {path}")

        model, processor = self._ensure_loaded()
        audio, sampling_rate = self.audio_loader(path, self.target_sampling_rate)
        inputs = self._process_audio(processor, audio, sampling_rate)
        inputs = self._move_inputs_to_model_device(inputs, model)
        features = self._extract_features(model, inputs)
        return _l2_normalize(_to_flat_vector(features))

    def _ensure_loaded(self) -> tuple[Any, Any]:
        if self._model is None:
            self._model = self.model_loader(self.model_name, self.local_files_only)
            if hasattr(self._model, "eval"):
                self._model.eval()
        if self._processor is None:
            self._processor = self.processor_loader(self.model_name, self.local_files_only)
        return self._model, self._processor

    def _extract_features(self, model: Any, inputs: dict[str, Any]) -> Any:
        try:
            import torch
        except Exception:
            return model.get_audio_features(**inputs)
        with torch.no_grad():
            return model.get_audio_features(**inputs)

    def _process_audio(self, processor: Any, audio: Any, sampling_rate: int) -> Any:
        """Handle both current and older transformers CLAP processor APIs."""

        try:
            return processor(audio=audio, sampling_rate=sampling_rate, return_tensors="pt")
        except TypeError:
            return processor(audios=audio, sampling_rate=sampling_rate, return_tensors="pt")

    def _move_inputs_to_model_device(self, inputs: Any, model: Any) -> Any:
        device = getattr(model, "device", None)
        if device is not None and hasattr(inputs, "to"):
            return inputs.to(device)
        return inputs

    def _load_model(self, model_name: str, local_files_only: bool) -> Any:
        from transformers import ClapModel

        return ClapModel.from_pretrained(model_name, local_files_only=local_files_only)

    def _load_processor(self, model_name: str, local_files_only: bool) -> Any:
        from transformers import ClapProcessor

        return ClapProcessor.from_pretrained(model_name, local_files_only=local_files_only)

    def _load_audio(self, path: Path, target_sampling_rate: int) -> tuple[Any, int]:
        import librosa

        audio, sampling_rate = librosa.load(str(path), sr=target_sampling_rate, mono=True)
        return audio, sampling_rate
