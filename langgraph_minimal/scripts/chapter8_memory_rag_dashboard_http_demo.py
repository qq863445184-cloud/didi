from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.chapter8_memory_rag_dashboard_demo import build_dashboard_demo


HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>第八章 Memory/RAG 管理页</title>
  <style>
    body { margin: 0; font-family: Arial, "Microsoft YaHei", sans-serif; background: #f6f7fb; color: #172033; }
    header { padding: 18px 28px; background: #1f2937; color: #fff; }
    main { display: grid; grid-template-columns: 360px 1fr; gap: 18px; padding: 18px; }
    section { background: #fff; border: 1px solid #d8deea; border-radius: 8px; padding: 16px; }
    h1 { margin: 0; font-size: 20px; }
    h2 { margin: 0 0 12px; font-size: 16px; }
    label { display: block; margin-top: 10px; font-size: 13px; color: #526078; }
    input, textarea { width: 100%; box-sizing: border-box; margin-top: 6px; padding: 9px; border: 1px solid #cbd3df; border-radius: 6px; font: inherit; }
    textarea { min-height: 78px; resize: vertical; }
    button { margin-top: 12px; margin-right: 8px; padding: 9px 12px; border: 0; border-radius: 6px; background: #2563eb; color: #fff; cursor: pointer; }
    button.secondary { background: #475569; }
    pre { min-height: 520px; white-space: pre-wrap; overflow: auto; background: #101827; color: #dbeafe; padding: 14px; border-radius: 8px; }
    .hint { margin-top: 8px; font-size: 12px; color: #64748b; }
  </style>
</head>
<body>
  <header><h1>第八章 Memory/RAG 管理页</h1></header>
  <main>
    <section>
      <h2>操作</h2>
      <button onclick="post('/api/ingest-demo')">导入业务多模态示例</button>
      <div class="hint">示例会生成发票图片和客服录音占位文件，并通过注入 OCR/ASR 文本入库。</div>

      <label>问题</label>
      <textarea id="question">Which invoice is tied to the refund request?</textarea>
      <button onclick="post('/api/ask', { question: value('question') })">提问</button>

      <label>记忆检索关键词</label>
      <input id="query" value="INV-2026-001 refund request" />
      <button onclick="post('/api/recall', { query: value('query') })">检索记忆</button>
      <button class="secondary" onclick="post('/api/inventory')">记忆库存</button>
      <button class="secondary" onclick="post('/api/trace')">Trace</button>
    </section>
    <section>
      <h2>输出</h2>
      <pre id="output">点击左侧按钮开始。</pre>
    </section>
  </main>
  <script>
    function value(id) { return document.getElementById(id).value; }
    async function post(url, payload = {}) {
      const output = document.getElementById('output');
      output.textContent = '请求中...';
      const body = new URLSearchParams(payload);
      const response = await fetch(url, { method: 'POST', body });
      output.textContent = await response.text();
    }
  </script>
</body>
</html>
"""


class DashboardHTTPHandler(BaseHTTPRequestHandler):
    dashboard = build_dashboard_demo()
    demo_dir = Path("memory_data") / "memory_rag_dashboard_http_demo"

    def do_GET(self) -> None:
        if self.path == "/":
            self._send(HTML, content_type="text/html; charset=utf-8")
            return
        self._send("Not found", status=404)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw_body = self.rfile.read(length).decode("utf-8")
        form = {key: values[0] for key, values in parse_qs(raw_body).items()}

        routes = {
            "/api/ingest-demo": self._ingest_demo,
            "/api/ask": lambda: self.dashboard.ask(form.get("question", "")),
            "/api/recall": lambda: self.dashboard.recall(form.get("query", "")),
            "/api/inventory": self.dashboard.memory_inventory,
            "/api/trace": self.dashboard.trace,
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
