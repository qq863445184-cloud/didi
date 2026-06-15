from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
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
        vl_rec_model_dir=str(model_dir.resolve()),
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
    )
    result = model.predict(str(file_path.resolve()))
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
        return _clean_text(value)
    if isinstance(value, (int, float, bool)):
        return ""
    if hasattr(value, "markdown"):
        extracted = _extract_text(value.markdown)
        if extracted:
            return extracted
    if isinstance(value, dict):
        parts = []
        # PaddleOCR-VL puts the final readable document in markdown.markdown_texts.
        # Individual layout blocks keep their OCR result in block_content; block_label
        # is only a category name such as "text" or "image" and must not be indexed.
        for key in ("markdown_texts", "markdown_text", "text", "rec_text", "content", "block_content"):
            if isinstance(value.get(key), str):
                text = value[key].strip()
                if text:
                    parts.append(text)
        if parts:
            return "\n".join(dict.fromkeys(parts)).strip()
        for key, item in value.items():
            if key in {
                "image",
                "img",
                "array",
                "pixels",
                "bbox",
                "box",
                "score",
                "ok",
                "block_label",
                "block_bbox",
                "block_polygon_points",
            }:
                continue
            extracted = _extract_text(item)
            if extracted:
                parts.append(extracted)
        return "\n".join(part for part in parts if part).strip()
    if isinstance(value, (list, tuple)):
        if _looks_like_numeric_array(value):
            return ""
        return "\n".join(part for item in value if (part := _extract_text(item))).strip()
    if hasattr(value, "json"):
        return _extract_text(value.json)
    if hasattr(value, "to_dict"):
        return _extract_text(value.to_dict())
    return ""


def _looks_like_numeric_array(value: Any) -> bool:
    if not isinstance(value, (list, tuple)):
        return False
    if not value:
        return True
    flattened = _flatten_limited(value, limit=32)
    if not flattened:
        return True
    return all(isinstance(item, (int, float, bool)) for item in flattened)


def _flatten_limited(value: Any, *, limit: int) -> list[Any]:
    items: list[Any] = []
    stack = [value]
    while stack and len(items) < limit:
        current = stack.pop()
        if isinstance(current, (list, tuple)):
            stack.extend(reversed(current))
        else:
            items.append(current)
    return items


def _clean_text(value: str) -> str:
    text = value.strip()
    if not text:
        return ""
    layout_labels = {
        "number",
        "footnote",
        "header",
        "header_image",
        "footer",
        "footer_image",
        "aside_text",
    }
    if text.lower() in layout_labels:
        return ""
    if re.search(r"[\\/].+\.(png|jpg|jpeg|webp|bmp|gif|wav|mp3|m4a|flac)$", text, re.IGNORECASE):
        return ""
    if re.fullmatch(r"[\[\]\d\s.,+-]+", text):
        return ""
    return text


if __name__ == "__main__":
    main()
