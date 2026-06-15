from app.my_memory_system.memory_rag_dashboard import MemoryRAGDashboard
from scripts.chapter8_business_multimodal_demo import build_business_multimodal_demo


def test_memory_rag_dashboard_drives_multimodal_rag_and_memory(tmp_path):
    invoice = tmp_path / "invoice_INV-2026-001.png"
    audio = tmp_path / "support_call_ORD-2026-778.wav"
    invoice.write_bytes(b"fake invoice image bytes")
    audio.write_bytes(b"fake support audio bytes")

    perception_tool, rag_tool, manager = build_business_multimodal_demo()
    dashboard = MemoryRAGDashboard(
        perception_tool=perception_tool,
        rag_tool=rag_tool,
        memory_manager=manager,
        rag_namespace="business_multimodal",
    )

    invoice_output = dashboard.ingest_file(
        str(invoice),
        description="Finance invoice screenshot.",
        importance=0.85,
    )
    audio_output = dashboard.ingest_file(
        str(audio),
        description="Customer support call recording.",
        importance=0.9,
    )
    answer_output = dashboard.ask("Which invoice is tied to the refund request?", limit=3)
    inventory_output = dashboard.memory_inventory()
    trace_output = dashboard.trace()

    assert "Perceptual memory saved" in invoice_output
    assert "Perceptual memory saved" in audio_output
    assert "INV-2026-001" in answer_output
    assert "检索证据" in answer_output
    assert "perceptual" in inventory_output
    assert "semantic" in inventory_output
    assert "perception.sync_rag" in trace_output
    assert "rag.ask" in trace_output


def test_memory_rag_dashboard_answers_from_uploaded_document_evidence(tmp_path):
    invoice = tmp_path / "invoice_INV-2026-001.png"
    audio = tmp_path / "support_call_ORD-2026-778.wav"
    patent = tmp_path / "cross_device_patent.md"
    invoice.write_bytes(b"fake invoice image bytes")
    audio.write_bytes(b"fake support audio bytes")
    patent.write_text(
        "\n".join(
            [
                "# 跨设备信息交互方案",
                "## 核心方案",
                "第一设备检测当前业务上下文，判断是否满足跨设备交互触发条件。",
                "第一设备基于统一交互协议发现第二设备，并进行身份认证和能力协商。",
                "系统按照最小必要数据同步策略建立临时会话，并在任务完成后自动终止。",
            ]
        ),
        encoding="utf-8",
    )

    perception_tool, rag_tool, manager = build_business_multimodal_demo()
    dashboard = MemoryRAGDashboard(
        perception_tool=perception_tool,
        rag_tool=rag_tool,
        memory_manager=manager,
        rag_namespace="business_multimodal",
    )

    dashboard.ingest_file(str(invoice), description="Finance invoice screenshot.", importance=0.85)
    dashboard.ingest_file(str(audio), description="Customer support call recording.", importance=0.9)
    dashboard.ingest_file(str(patent), description="Cross-device patent draft.", importance=0.8)

    answer_output = dashboard.ask("跨设备信息交互的核心方案是什么？", limit=5)

    assert "核心方案" in answer_output
    assert "统一交互协议" in answer_output
    assert "cross_device_patent.md" in answer_output
    assert "INV-2026-001" not in answer_output
    assert "ORD-2026-778" not in answer_output
    assert "已从发票图片和客服录音中确认" not in answer_output


def test_memory_rag_dashboard_answers_protocol_field_questions(tmp_path):
    patent = tmp_path / "cross_device_patent.md"
    patent.write_text(
        "\n".join(
            [
                "# 跨设备信息交互方案",
                "## 核心方案",
                "第一设备检测当前业务上下文，判断是否满足跨设备交互触发条件。",
                "第一设备基于统一交互协议发现第二设备，并进行身份认证和能力协商。",
                "## 统一协议字段",
                "统一交互协议可包括：",
                "- task_id：任务标识；",
                "- context_type：上下文类型；",
                "- device_capability：设备能力描述；",
                "- data_scope：同步数据范围；",
                "- permission_scope：权限范围；",
                "- lifecycle_policy：会话生命周期策略；",
                "- stop_condition：停止条件。",
            ]
        ),
        encoding="utf-8",
    )

    perception_tool, rag_tool, manager = build_business_multimodal_demo()
    dashboard = MemoryRAGDashboard(
        perception_tool=perception_tool,
        rag_tool=rag_tool,
        memory_manager=manager,
        rag_namespace="business_multimodal",
    )

    dashboard.ingest_file(str(patent), description="Cross-device patent draft.", importance=0.8)

    answer_output = dashboard.ask("统一交互协议可包括哪些？", limit=5)

    assert "- task_id：任务标识" in answer_output
    assert "- context_type：上下文类型" in answer_output
    assert "统一交互协议可包括" in answer_output
    assert "task_id" in answer_output
    assert "data_scope" in answer_output
    assert "stop_condition" in answer_output
    assert "根据检索到的" not in answer_output
    assert "核心方案" not in answer_output
    assert "rag.ask" in dashboard.trace()
