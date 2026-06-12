from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def build_health_report(
    *,
    model_root: str | Path | None = None,
    check_services: bool = False,
    timeout_seconds: float = 2.0,
) -> dict[str, Any]:
    """Return a lightweight readiness report for the chapter 8 backends.

    这个检查只验证“可用条件”而不加载重模型：模型目录、关键权重文件、依赖包、
    Qdrant/Neo4j 配置以及可选服务连通性。真实 OCR/ASR/向量推理放到 smoke
    demo 里做，避免一次健康检查就占用大量内存。
    """

    load_dotenv(PROJECT_ROOT / ".env", override=False)
    root = Path(model_root) if model_root is not None else PROJECT_ROOT / "models"
    embed_model = Path(os.getenv("EMBED_MODEL_NAME") or root / "all-MiniLM-L6-v2")

    multimodal_runtime = _multimodal_runtime_status()
    report = {
        "project_root": str(PROJECT_ROOT),
        "packages": {
            "markitdown": _package_status("markitdown"),
            "spacy": _package_status("spacy"),
            "sentence_transformers": _package_status("sentence_transformers"),
            "qdrant_client": _package_status("qdrant_client"),
            "neo4j": _package_status("neo4j"),
            "funasr": _package_status("funasr"),
            "paddleocr": _package_status("paddleocr"),
            "transformers": _package_status("transformers"),
            "librosa": _package_status("librosa"),
        },
        "models": {
            "text_embedding": _model_status(
                embed_model,
                required_files=["model.safetensors", "pytorch_model.bin"],
                any_required=True,
            ),
            "sensevoice_asr": _model_status(
                root / "iic" / "SenseVoiceSmall",
                required_files=["model.pt", "config.yaml"],
            ),
            "paddleocr_vl": _model_status(
                _resolve_existing(
                    root / "PaddlePaddle" / "PaddleOCR-VL-1.6",
                    root / "PaddlePaddle" / "PaddleOCR-VL-1___6",
                ),
                required_files=["model.safetensors", "config.json"],
            ),
            "clip_image": _optional_transformers_model_status(
                os.getenv("CLIP_MODEL_NAME", "openai/clip-vit-base-patch32"),
                required_packages=["transformers"],
                note="optional; lazy loaded by ClipImageEmbedder",
            ),
            "clap_audio": _optional_transformers_model_status(
                os.getenv("CLAP_MODEL_NAME", "laion/clap-htsat-unfused"),
                required_packages=["transformers", "librosa"],
                note="optional; lazy loaded by ClapAudioEmbedder",
            ),
        },
        "services": {
            "qdrant": _qdrant_status(check_services=check_services, timeout_seconds=timeout_seconds),
            "neo4j": _neo4j_status(check_services=check_services, timeout_seconds=timeout_seconds),
        },
        "multimodal_runtime": multimodal_runtime,
    }
    report["ready"] = _overall_ready(report)
    return report


def _package_status(module_name: str) -> dict[str, Any]:
    return {"installed": importlib.util.find_spec(module_name) is not None}


def _multimodal_runtime_status() -> dict[str, Any]:
    """Check an optional external Python used for heavy OCR/ASR dependencies.

    主教程环境可能是 Python 3.13，而 FunASR/PaddleOCR 这类重型多模态依赖
    更适合放在独立 Python 3.11 环境里。这里仅检查 import 可用性，不加载模型。
    """

    python_path = os.getenv("MULTIMODAL_PYTHON") or str(PROJECT_ROOT / ".venv-asr" / "Scripts" / "python.exe")
    path = Path(python_path)
    status: dict[str, Any] = {
        "configured": bool(python_path),
        "python": python_path,
        "exists": path.exists(),
        "packages": {
            "funasr": {"installed": False},
            "paddleocr": {"installed": False},
        },
        "ready": False,
    }
    if not path.exists():
        return status

    for module_name in status["packages"]:
        status["packages"][module_name] = _external_python_package_status(path, module_name)
    status["ready"] = all(item["installed"] for item in status["packages"].values())
    return status


def _external_python_package_status(python_path: Path, module_name: str) -> dict[str, Any]:
    env = os.environ.copy()
    code = f"import importlib.util; raise SystemExit(0 if importlib.util.find_spec({module_name!r}) else 1)"
    try:
        completed = subprocess.run(
            [str(python_path), "-c", code],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=8,
            env=env,
            check=False,
        )
        return {"installed": completed.returncode == 0}
    except Exception as exc:
        return {"installed": False, "error": str(exc)}


def _model_status(
    path: Path,
    *,
    required_files: list[str],
    any_required: bool = False,
) -> dict[str, Any]:
    exists = path.exists() and path.is_dir()
    file_states = {name: (path / name).exists() for name in required_files}
    has_required = any(file_states.values()) if any_required else all(file_states.values())
    return {
        "path": str(path),
        "exists": exists,
        "files": file_states,
        "ready": bool(exists and has_required),
    }


def _resolve_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def _optional_transformers_model_status(
    model_name: str,
    *,
    required_packages: list[str],
    note: str,
) -> dict[str, Any]:
    packages = {name: _package_status(name)["installed"] for name in required_packages}
    package_ready = all(packages.values())
    status: dict[str, Any] = {
        "optional": True,
        "ready": False,
        "model_name": model_name,
        "packages": packages,
        "local_files_only": True,
        "note": note,
    }
    if not package_ready:
        return status

    path = Path(model_name)
    if path.exists():
        status["path"] = str(path)
        status["ready"] = (path / "config.json").exists()
        return status

    try:
        from transformers.utils import cached_file

        cached_path = cached_file(model_name, "config.json", local_files_only=True)
    except Exception as exc:
        status["cache_error"] = str(exc)
        return status

    status["cached_config"] = str(cached_path)
    status["ready"] = bool(cached_path)
    return status


def _qdrant_status(*, check_services: bool, timeout_seconds: float) -> dict[str, Any]:
    url = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
    status: dict[str, Any] = {
        "configured": bool(url),
        "url": url,
        "api_key": _mask_secret(os.getenv("QDRANT_API_KEY")),
        "reachable": None,
    }
    if check_services and url:
        status["reachable"] = _http_get_ok(f"{url.rstrip('/')}/collections", timeout_seconds)
    return status


def _neo4j_status(*, check_services: bool, timeout_seconds: float) -> dict[str, Any]:
    uri = os.getenv("NEO4J_URI", "")
    username = os.getenv("NEO4J_USERNAME", "")
    password = os.getenv("NEO4J_PASSWORD", "")
    status: dict[str, Any] = {
        "configured": bool(uri and username and password),
        "uri": uri,
        "username": username,
        "password": _mask_secret(password),
        "database": os.getenv("NEO4J_DATABASE", "neo4j"),
        "reachable": None,
    }
    if check_services and status["configured"]:
        status["reachable"] = _neo4j_can_connect(timeout_seconds=timeout_seconds)
    return status


def _http_get_ok(url: str, timeout_seconds: float) -> bool:
    try:
        request = Request(url, headers={"Accept": "application/json"})
        with urlopen(request, timeout=timeout_seconds) as response:
            return 200 <= response.status < 300
    except Exception:
        return False


def _neo4j_can_connect(*, timeout_seconds: float) -> bool:
    try:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(
            os.environ["NEO4J_URI"],
            auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
            connection_timeout=timeout_seconds,
        )
        with driver:
            driver.verify_connectivity()
        return True
    except Exception:
        return False


def _mask_secret(value: str | None) -> str | None:
    if not value:
        return None
    return "***"


def _overall_ready(report: dict[str, Any]) -> bool:
    required_models = ["text_embedding", "sensevoice_asr", "paddleocr_vl"]
    required_packages = ["markitdown", "spacy", "sentence_transformers", "qdrant_client", "neo4j"]
    core_ready = all(report["models"][name]["ready"] for name in required_models) and all(
        report["packages"][name]["installed"] for name in required_packages
    )
    heavy_packages_ready = (
        report["packages"]["funasr"]["installed"]
        and report["packages"]["paddleocr"]["installed"]
    ) or report["multimodal_runtime"]["ready"]
    return bool(core_ready and heavy_packages_ready)


def main() -> None:
    parser = argparse.ArgumentParser(description="Chapter 8 backend and model readiness check")
    parser.add_argument("--model-root", default=str(PROJECT_ROOT / "models"))
    parser.add_argument("--check-services", action="store_true")
    parser.add_argument("--timeout", type=float, default=2.0)
    args = parser.parse_args()

    report = build_health_report(
        model_root=args.model_root,
        check_services=args.check_services,
        timeout_seconds=args.timeout,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
