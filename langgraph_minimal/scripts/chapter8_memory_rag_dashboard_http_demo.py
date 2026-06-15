from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from email.parser import BytesParser
from email.policy import default
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.chapter8_memory_rag_dashboard_demo import (
    build_dashboard_demo,
    build_persistent_dashboard_demo,
)


HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>第八章 Memory/RAG 管理页</title>
  <style>
    body { margin: 0; font-family: Arial, "Microsoft YaHei", sans-serif; background: #f6f7fb; color: #172033; }
    header { padding: 18px 28px; background: #1f2937; color: #fff; }
    main { display: grid; grid-template-columns: minmax(300px, 360px) minmax(0, 1fr); gap: 18px; padding: 18px; }
    section { background: #fff; border: 1px solid #d8deea; border-radius: 8px; padding: 16px; }
    h1 { margin: 0; font-size: 20px; }
    h2 { margin: 0 0 12px; font-size: 16px; }
    label { display: block; margin-top: 10px; font-size: 13px; color: #526078; }
    input, textarea { width: 100%; box-sizing: border-box; margin-top: 6px; padding: 9px; border: 1px solid #cbd3df; border-radius: 6px; font: inherit; }
    textarea { min-height: 78px; resize: vertical; }
    button { margin-top: 12px; margin-right: 8px; padding: 9px 12px; border: 0; border-radius: 6px; background: #2563eb; color: #fff; cursor: pointer; }
    button.secondary { background: #475569; }
    pre { min-height: 520px; white-space: pre-wrap; overflow: auto; overflow-wrap: anywhere; background: #101827; color: #dbeafe; padding: 14px; border-radius: 8px; }
    .hint { margin-top: 8px; font-size: 12px; color: #64748b; }
    @media (max-width: 820px) {
      main { grid-template-columns: 1fr; padding: 12px; }
      pre { min-height: 360px; }
    }
  </style>
</head>
<body>
  <header><h1>第八章 Memory/RAG 管理页</h1></header>
  <main>
    <section>
      <h2>操作</h2>
      <button type="button" onclick="window.submitDashboardAction('/api/ingest-demo')">导入业务多模态示例</button>
      <div class="hint">示例按钮使用固定样例抽取文本；上传真实图片/音频时会走 OCR/ASR 抽取文本。</div>

      <label>上传其他文件</label>
      <input id="uploadFile" type="file" />
      <input id="uploadDescription" value="User uploaded file for memory/RAG demo." />
      <button type="button" onclick="window.uploadDashboardFile()">上传并感知入库</button>
      <div class="hint">支持文本、Markdown、图片、音频等；图片/音频会先抽取文本，再同步到 RAG。</div>

      <label>问题</label>
      <textarea id="question">Which invoice is tied to the refund request?</textarea>
      <button type="button" onclick="window.askDashboardQuestion()">提问</button>

      <label>记忆检索关键词</label>
      <input id="query" value="INV-2026-001 refund request" />
      <button type="button" onclick="window.recallDashboardMemory()">检索记忆</button>
      <button type="button" class="secondary" onclick="window.submitDashboardAction('/api/inventory')">记忆库存</button>
      <button type="button" class="secondary" onclick="window.submitDashboardAction('/api/trace')">Trace</button>
      <button type="button" class="secondary" onclick="window.submitDashboardAction('/api/reset')">重置当前知识库</button>
    </section>
    <section>
      <h2>输出</h2>
      <pre id="output">点击左侧按钮开始。</pre>
    </section>
  </main>
  <script>
    function value(id) { return document.getElementById(id).value; }
    window.askDashboardQuestion = function() {
      return window.submitDashboardAction('/api/ask', { question: value('question') });
    }
    window.recallDashboardMemory = function() {
      return window.submitDashboardAction('/api/recall', { query: value('query') });
    }
    window.submitDashboardAction = async function(url, payload = {}) {
      const output = document.getElementById('output');
      const start = Date.now();
      output.textContent = `请求中... (${new Date().toLocaleTimeString()})`;
      try {
        const body = new URLSearchParams(payload);
        const response = await fetch(url, { method: 'POST', body });
        const text = await response.text();
        const elapsed = ((Date.now() - start) / 1000).toFixed(1);
        output.textContent = `[${response.status}] ${elapsed}s\n\n${text}`;
      } catch (error) {
        output.textContent = `请求失败：${error.message || error}`;
      }
    }
    window.uploadDashboardFile = async function() {
      const output = document.getElementById('output');
      const fileInput = document.getElementById('uploadFile');
      if (!fileInput.files.length) {
        output.textContent = '请先选择文件。';
        return;
      }
      const start = Date.now();
      output.textContent = `上传中... (${new Date().toLocaleTimeString()})`;
      const body = new FormData();
      body.append('file', fileInput.files[0]);
      body.append('description', value('uploadDescription'));
      body.append('importance', '0.75');
      try {
        const response = await fetch('/api/upload', { method: 'POST', body });
        const text = await response.text();
        const elapsed = ((Date.now() - start) / 1000).toFixed(1);
        output.textContent = `[${response.status}] ${elapsed}s\n\n${text}`;
      } catch (error) {
        output.textContent = `上传失败：${error.message || error}`;
      }
    }
  </script>
</body>
</html>
"""


def build_http_dashboard():
    """Create the dashboard used by the lightweight HTTP page.

    默认保持上一版的内存 demo，保证本地页面打开就能用。设置
    CHAPTER8_DASHBOARD_PERSISTENT=1 后，会切到 SQLite/JSON 持久化版本；
    再设置 CHAPTER8_DASHBOARD_EXTERNAL_BACKENDS=1 时尝试接 Qdrant/Neo4j。
    """

    persistent = _env_flag("CHAPTER8_DASHBOARD_PERSISTENT", default=False)
    real_llm = _env_flag("CHAPTER8_DASHBOARD_REAL_LLM", default=True)
    real_multimodal = _env_flag("CHAPTER8_DASHBOARD_REAL_MULTIMODAL", default=True)
    if not persistent:
        return build_dashboard_demo(
            prefer_real_llm=real_llm,
            prefer_real_multimodal=real_multimodal,
        )

    return build_persistent_dashboard_demo(
        data_dir=os.getenv("CHAPTER8_DASHBOARD_DATA_DIR"),
        prefer_real_llm=real_llm,
        prefer_real_multimodal=real_multimodal,
        prefer_external_backends=_env_flag("CHAPTER8_DASHBOARD_EXTERNAL_BACKENDS", default=False),
        strict_backends=_env_flag("CHAPTER8_DASHBOARD_STRICT_BACKENDS", default=False),
    )


def _env_flag(name: str, *, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class DashboardHTTPHandler(BaseHTTPRequestHandler):
    dashboard = build_http_dashboard()
    demo_dir = Path("memory_data") / "memory_rag_dashboard_http_demo"
    upload_dir = Path("memory_data") / "memory_rag_dashboard_uploads"

    def do_GET(self) -> None:
        if self.path == "/":
            self._send(HTML, content_type="text/html; charset=utf-8")
            return
        self._send("Not found", status=404)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw_payload = self.rfile.read(length)
        if self.path == "/api/upload":
            self._handle_upload(raw_payload)
            return
        raw_body = raw_payload.decode("utf-8")
        form = {key: values[0] for key, values in parse_qs(raw_body).items()}

        routes = {
            "/api/ingest-demo": self._ingest_demo,
            "/api/ask": lambda: self.dashboard.ask(form.get("question", "")),
            "/api/recall": lambda: self.dashboard.recall(form.get("query", "")),
            "/api/inventory": self.dashboard.memory_inventory,
            "/api/trace": self.dashboard.trace,
            "/api/reset": self._reset_dashboard,
        }
        action = routes.get(self.path)
        if action is None:
            self._send("Not found", status=404)
            return

        try:
            self._send(str(action()))
        except Exception as exc:
            self._send(f"Error: {exc}", status=500)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _ingest_demo(self) -> str:
        self.demo_dir.mkdir(parents=True, exist_ok=True)
        invoice = self.demo_dir / "invoice_INV-2026-001.png"
        audio = self.demo_dir / "support_call_ORD-2026-778.wav"
        invoice.write_bytes(b"fake invoice image bytes")
        audio.write_bytes(b"fake support audio bytes")
        perception_tool = self.dashboard.perception_tool
        if perception_tool is None:
            return "未配置感知工具，无法导入示例。"

        # 示例文件是占位字节，不适合触发真实重模型；真实 OCR/ASR 留给用户上传文件时执行。
        old_image_ocr = perception_tool.image_ocr
        old_audio_asr = perception_tool.audio_asr
        perception_tool.image_ocr = self._demo_image_ocr
        perception_tool.audio_asr = self._demo_audio_asr
        try:
            invoice_output = self.dashboard.ingest_file(
                invoice,
                "Finance invoice screenshot from the refund workflow.",
                0.85,
            )
            audio_output = self.dashboard.ingest_file(
                audio,
                "Customer support call recording linked to refund workflow.",
                0.9,
            )
            return "\n\n".join([invoice_output, audio_output])
        finally:
            perception_tool.image_ocr = old_image_ocr
            perception_tool.audio_asr = old_audio_asr

    def _demo_image_ocr(self, _path: Path) -> str:
        return (
            "Invoice No: INV-2026-001. Vendor: Cloud Training Ltd. "
            "Total amount: 1280 CNY. Payment status: paid."
        )

    def _demo_audio_asr(self, _path: Path) -> str:
        return (
            "Support call transcript: customer made a refund request for order "
            "ORD-2026-778 and referenced invoice INV-2026-001."
        )

    def _reset_dashboard(self) -> str:
        """Reset the in-memory demo stack without deleting uploaded files.

        The HTTP demo keeps RAG vectors and memories in process memory.  A reset
        gives manual testers a clean knowledge base when previous smoke tests or
        uploads start affecting retrieval results.
        """

        type(self).dashboard = build_http_dashboard()
        return "当前内存知识库已重置。已上传的文件仍保留在 memory_data 目录，需要重新上传/入库后才能再次检索。"

    def _handle_upload(self, raw_payload: bytes) -> None:
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            self._send("Error: upload must use multipart/form-data", status=400)
            return

        message = BytesParser(policy=default).parsebytes(
            (
                f"Content-Type: {content_type}\r\n"
                "MIME-Version: 1.0\r\n\r\n"
            ).encode("utf-8")
            + raw_payload
        )
        fields: dict[str, str] = {}
        uploaded_file: tuple[str, bytes] | None = None
        for part in message.iter_parts():
            name = part.get_param("name", header="content-disposition")
            filename = part.get_filename()
            payload = part.get_payload(decode=True) or b""
            if filename:
                uploaded_file = (filename, payload)
            elif name:
                fields[name] = payload.decode("utf-8", errors="ignore")

        if uploaded_file is None:
            self._send("Error: no file uploaded", status=400)
            return

        original_name, data = uploaded_file
        safe_name = self._safe_filename(original_name)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        target = self.upload_dir / f"{uuid.uuid4().hex}_{safe_name}"
        target.write_bytes(data)

        result = self.dashboard.ingest_file(
            target,
            fields.get("description", ""),
            float(fields.get("importance", "0.75") or 0.75),
        )
        self._send(
            "\n".join(
                [
                    f"Uploaded file saved: {target}",
                    "",
                    result,
                ]
            )
        )

    def _safe_filename(self, filename: str) -> str:
        keep = []
        for char in Path(filename).name:
            keep.append(char if char.isalnum() or char in {".", "_", "-"} else "_")
        return "".join(keep).strip("._") or "uploaded_file"

    def _send(self, text: str, *, status: int = 200, content_type: str = "text/plain; charset=utf-8") -> None:
        data = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch no-dependency Memory/RAG dashboard demo.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7868)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), DashboardHTTPHandler)
    print(f"Memory/RAG dashboard: http://{args.host}:{args.port}/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
