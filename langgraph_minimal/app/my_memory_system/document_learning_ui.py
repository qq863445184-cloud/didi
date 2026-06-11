from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .document_learning_assistant import DocumentLearningAssistant


class DocumentLearningUI:
    """Thin UI callback layer for the chapter 8 learning assistant.

    Gradio 只负责输入输出控件；真正的 RAG、记忆写入和报告生成都留在
    DocumentLearningAssistant。这样即使没有安装 Gradio，回调逻辑也能测试。
    """

    def __init__(self, assistant: DocumentLearningAssistant) -> None:
        self.assistant = assistant

    def load_document(self, file_path: str | Path | None) -> str:
        if not file_path:
            return "请先选择文档。"
        result = self.assistant.load_document(file_path)
        return result.raw_output

    def ask(self, question: str) -> str:
        if not question.strip():
            return "请输入问题。"
        result = self.assistant.ask(question)
        lines = [result.answer]
        if result.references:
            lines.append("")
            lines.append("参考来源：")
            lines.extend(result.references)
        return "\n".join(lines)

    def add_note(self, note: str) -> str:
        if not note.strip():
            return "请输入学习笔记。"
        return self.assistant.add_note(note)

    def recall(self, query: str) -> str:
        if not query.strip():
            return "请输入要回顾的关键词。"
        return self.assistant.recall(query)

    def generate_report(self) -> str:
        report = self.assistant.generate_report()
        return json.dumps(report, ensure_ascii=False, indent=2)

    def trace(self) -> str:
        return json.dumps(self.assistant.trace_events, ensure_ascii=False, indent=2)


def build_gradio_app(assistant: DocumentLearningAssistant) -> Any:
    """Build the optional Gradio interface used by the chapter 8 demo."""

    try:
        import gradio as gr
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Gradio 未安装。请先执行 pip install gradio 后再启动 Web UI。"
        ) from exc

    ui = DocumentLearningUI(assistant)
    with gr.Blocks(title="第八章 文档学习助手") as demo:
        gr.Markdown("# 第八章 文档学习助手")
        with gr.Row():
            file_input = gr.File(label="上传文档", type="filepath")
            load_output = gr.Textbox(label="加载结果", lines=5)
        load_button = gr.Button("加载文档")

        question_input = gr.Textbox(label="问题", placeholder="例如：RAG 学习流程包括什么？")
        answer_output = gr.Textbox(label="回答", lines=8)
        ask_button = gr.Button("提问")

        note_input = gr.Textbox(label="学习笔记")
        note_output = gr.Textbox(label="笔记结果", lines=3)
        note_button = gr.Button("保存笔记")

        recall_input = gr.Textbox(label="回顾关键词")
        recall_output = gr.Textbox(label="回顾结果", lines=6)
        recall_button = gr.Button("回顾")

        report_output = gr.Code(label="学习报告", language="json")
        trace_output = gr.Code(label="Trace", language="json")
        with gr.Row():
            report_button = gr.Button("生成报告")
            trace_button = gr.Button("查看 Trace")

        load_button.click(ui.load_document, inputs=file_input, outputs=load_output)
        ask_button.click(ui.ask, inputs=question_input, outputs=answer_output)
        note_button.click(ui.add_note, inputs=note_input, outputs=note_output)
        recall_button.click(ui.recall, inputs=recall_input, outputs=recall_output)
        report_button.click(ui.generate_report, outputs=report_output)
        trace_button.click(ui.trace, outputs=trace_output)

    return demo
