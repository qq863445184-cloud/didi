from pathlib import Path

import pytest

from app.my_memory_system.multimodal_encoders import (
    ClapAudioEmbedder,
    ClipImageEmbedder,
    _l2_normalize,
)


class FakeTensor:
    def __init__(self, values):
        self.values = values

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return [self.values]


class FakeModelOutput:
    def __init__(self, values):
        self.pooler_output = FakeTensor(values)


class FakeClipModel:
    def __init__(self):
        self.calls = 0

    def get_image_features(self, **kwargs):
        self.calls += 1
        return FakeTensor([3.0, 4.0])


class FakeClipProcessor:
    def __call__(self, *, images, return_tensors):
        return {"pixel_values": "fake-pixels"}


class FakeClapModel:
    def get_audio_features(self, **kwargs):
        return FakeTensor([0.0, 5.0])


class FakeClapProcessor:
    def __call__(self, *, audios, sampling_rate, return_tensors):
        return {"input_features": "fake-audio"}


class FakeNewClapProcessor:
    def __init__(self):
        self.received_audio = None

    def __call__(self, *, audio, sampling_rate, return_tensors):
        self.received_audio = audio
        return {"input_features": "fake-audio"}


def test_l2_normalize_returns_unit_vector():
    assert _l2_normalize([3.0, 4.0]) == [0.6, 0.8]


def test_l2_normalize_rejects_zero_vector():
    with pytest.raises(ValueError, match="zero vector"):
        _l2_normalize([0.0, 0.0])


def test_clip_image_embedder_accepts_transformers_model_output(tmp_path):
    image = tmp_path / "chart.png"
    image.write_bytes(b"fake image bytes")

    class OutputClipModel:
        def get_image_features(self, **kwargs):
            return FakeModelOutput([6.0, 8.0])

    embedder = ClipImageEmbedder(
        model_loader=lambda model_name, local_files_only: OutputClipModel(),
        processor_loader=lambda model_name, local_files_only: FakeClipProcessor(),
        image_loader=lambda path: "fake-image",
    )

    assert embedder(image) == [0.6, 0.8]


def test_clip_image_embedder_loads_lazily_and_returns_normalized_vector(tmp_path):
    image = tmp_path / "chart.png"
    image.write_bytes(b"fake image bytes")
    fake_model = FakeClipModel()
    load_calls = []

    embedder = ClipImageEmbedder(
        model_loader=lambda model_name, local_files_only: load_calls.append(model_name)
        or fake_model,
        processor_loader=lambda model_name, local_files_only: FakeClipProcessor(),
        image_loader=lambda path: "fake-image",
    )

    assert load_calls == []
    assert embedder(image) == [0.6, 0.8]
    assert embedder(image) == [0.6, 0.8]
    assert load_calls == ["openai/clip-vit-base-patch32"]
    assert fake_model.calls == 2


def test_clip_image_embedder_returns_clear_error_for_missing_file(tmp_path):
    embedder = ClipImageEmbedder(
        model_loader=lambda model_name, local_files_only: FakeClipModel(),
        processor_loader=lambda model_name, local_files_only: FakeClipProcessor(),
        image_loader=lambda path: "fake-image",
    )

    with pytest.raises(FileNotFoundError, match="Image file does not exist"):
        embedder(tmp_path / "missing.png")


def test_clap_audio_embedder_loads_lazily_and_returns_normalized_vector(tmp_path):
    audio = tmp_path / "meeting.wav"
    audio.write_bytes(b"fake audio bytes")
    load_calls = []

    embedder = ClapAudioEmbedder(
        model_loader=lambda model_name, local_files_only: load_calls.append(model_name)
        or FakeClapModel(),
        processor_loader=lambda model_name, local_files_only: FakeClapProcessor(),
        audio_loader=lambda path, target_sampling_rate: ([0.1, 0.2], target_sampling_rate),
    )

    assert load_calls == []
    assert embedder(audio) == [0.0, 1.0]
    assert load_calls == ["laion/clap-htsat-unfused"]


def test_clap_audio_embedder_accepts_new_transformers_audio_argument(tmp_path):
    audio = tmp_path / "meeting.wav"
    audio.write_bytes(b"fake audio bytes")
    processor = FakeNewClapProcessor()

    embedder = ClapAudioEmbedder(
        model_loader=lambda model_name, local_files_only: FakeClapModel(),
        processor_loader=lambda model_name, local_files_only: processor,
        audio_loader=lambda path, target_sampling_rate: ([0.1, 0.2], target_sampling_rate),
    )

    assert embedder(audio) == [0.0, 1.0]
    assert processor.received_audio == [0.1, 0.2]


def test_clap_audio_embedder_returns_clear_error_for_missing_file(tmp_path):
    embedder = ClapAudioEmbedder(
        model_loader=lambda model_name, local_files_only: FakeClapModel(),
        processor_loader=lambda model_name, local_files_only: FakeClapProcessor(),
        audio_loader=lambda path, target_sampling_rate: ([0.1], target_sampling_rate),
    )

    with pytest.raises(FileNotFoundError, match="Audio file does not exist"):
        embedder(tmp_path / "missing.wav")
