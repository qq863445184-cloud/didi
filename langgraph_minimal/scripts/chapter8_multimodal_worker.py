from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Chapter 8 external multimodal runtime worker")
    parser.add_argument("action", choices=["check", "asr", "ocr"])
    parser.add_argument("--file", default="")
    parser.add_argument("--model-dir", default="")
    args = parser.parse_args()

    try:
        if args.action == "check":
            payload = run_check()
        elif args.action == "asr":
            payload = {"text": run_asr(Path(args.file), Path(args.model_dir))}
        else:
            payload = {"text": run_ocr(Path(args.file), Path(args.model_dir))}
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        raise SystemExit(1) from exc

    print(json.dumps({"ok": True, **payload}, ensure_ascii=False))


def run_check() -> dict[str, Any]:
    return {
        "packages": {
            "funasr": importlib.util.find_spec("funasr") is not None,
            "paddleocr": importlib.util.find_spec("paddleocr") is not None,
        }
    }


def run_asr(file_path: Path, model_dir: Path) -> str:
    _ensure_file(file_path)
    if os.getenv("CHAPTER8_FAKE_MULTIMODAL"):
        return f"fake ASR text from {file_path.name}"

    from funasr import AutoModel

    model = AutoModel(model=str(model_dir))
    result = model.generate(
        input=str(file_path),
        language="auto",
        use_itn=True,
        batch_size_s=60,
    )
    return _extract_text(result)


def run_ocr(file_path: Path, model_dir: Path) -> str:
    _ensure_file(file_path)
    if os.getenv("CHAPTER8_FAKE_MULTIMODAL"):
        return f"fake OCR text from {file_path.name}"

    from paddleocr import PaddleOCRVL

    model = PaddleOCRVL(
        pipeline_version="v1.6",
        vl_rec_model_dir=str(model_dir),
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
    )
    result = model.predict(str(file_path))
    return _extract_text(result)


def _ensure_file(file_path: Path) -> None:
    if not file_path.exists():
        raise FileNotFoundError(f"input file does not exist: {file_path}")
    if not file_path.is_file():
        raise ValueError(f"input path is not a file: {file_path}")


def _extract_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        parts = []
        for key in ("text", "rec_text", "content", "markdown_text"):
            if isinstance(value.get(key), str):
                parts.append(value[key])
        parts.extend(_extract_text(item) for item in value.values())
        return "\n".join(part for part in parts if part).strip()
    if isinstance(value, (list, tuple)):
        return "\n".join(part for item in value if (part := _extract_text(item))).strip()
    if hasattr(value, "json"):
        return _extract_text(value.json)
    if hasattr(value, "to_dict"):
        return _extract_text(value.to_dict())
    return str(value).strip()


if __name__ == "__main__":
    main()
