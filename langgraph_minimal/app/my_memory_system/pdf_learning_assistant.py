from __future__ import annotations

from pathlib import Path
from typing import Any

from .document_learning_assistant import DocumentLearningAssistant


class PDFLearningAssistant(DocumentLearningAssistant):
    """Chapter 8 compatible facade for the document learning assistant.

    教程案例使用 PDFLearningAssistant、load_pdf、ask_question 这些命名。
    我们的底层实现已经支持任意文档，因此这里保留教程接口，避免使用者
    在照着第八章敲代码时还要做一层心智转换。
    """

    def load_pdf(self, pdf_path: str | Path, **kwargs: Any) -> str:
        result = self.load_document(pdf_path, **kwargs)
        return result.raw_output

    def ask_question(self, question: str, **kwargs: Any) -> str:
        result = self.ask_auto(question, **kwargs)
        lines = [result.answer]
        if result.references:
            lines.append("")
            lines.append("参考来源：")
            lines.extend(result.references)
        return "\n".join(lines)
