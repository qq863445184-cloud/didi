import json
import uuid
from typing import Any, Literal

from fastapi import FastAPI, WebSocket
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from app.runner import answer_events, answer_question

app = FastAPI(title="LangGraph Minimal Agent Protocol Server")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    mode: Literal["auto", "general", "rag", "plan"] = "auto"
    session_id: str = "default"


class AskResponse(BaseModel):
    answer: str
    mode: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask")
def ask(request: AskRequest) -> AskResponse:
    return AskResponse(
        answer=answer_question(request.question, request.mode, session_id=request.session_id),
        mode=request.mode,
    )


@app.post("/rag")
def rag(request: AskRequest) -> AskResponse:
    return AskResponse(
        answer=answer_question(request.question, "rag", session_id=request.session_id),
        mode="rag",
    )


@app.get("/events/ask")
def ask_events(
    question: str,
    mode: Literal["auto", "general", "rag", "plan"] = "auto",
    session_id: str = "default",
):
    def stream():
        for event in answer_events(question, mode, session_id=session_id):
            yield f"event: {event['event']}\n"
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    while True:
        payload = await websocket.receive_json()
        question = payload.get("question", "")
        mode = payload.get("mode", "auto")
        session_id = payload.get("session_id", "default")
        for event in answer_events(question, mode, session_id=session_id):
            await websocket.send_json(event)


@app.get("/.well-known/agent-card.json")
def agent_card() -> dict[str, Any]:
    return {
        "name": "langgraph-minimal-agent",
        "description": "LangGraph agent with tools, memory, and advanced local RAG.",
        "version": "0.1.0",
        "url": "http://localhost:8000/a2a",
        "capabilities": {
            "streaming": True,
            "modes": ["auto", "general", "rag", "plan"],
        },
        "skills": [
            {
                "id": "ask",
                "name": "Ask",
                "description": "Answer a user question using automatic routing.",
            },
            {
                "id": "plan",
                "name": "Plan",
                "description": "Create a coding plan without editing files.",
            },
            {
                "id": "rag",
                "name": "Project RAG",
                "description": "Answer project questions with cited retrieval evidence.",
            },
        ],
    }


@app.post("/a2a")
def a2a_rpc(payload: dict[str, Any]) -> JSONResponse:
    request_id = payload.get("id")
    method = payload.get("method")
    params = payload.get("params", {})

    if method not in {"message/send", "message.send", "tasks/send", "tasks.send"}:
        return _jsonrpc_error(request_id, -32601, f"Unsupported method: {method}")

    question = _extract_a2a_text(params)
    mode = params.get("mode", "auto")
    session_id = params.get("session_id", "default")
    task_id = params.get("id") or str(uuid.uuid4())

    try:
        answer = answer_question(question, mode, session_id=session_id)
    except Exception as exc:
        return _jsonrpc_error(request_id, -32000, str(exc))

    return JSONResponse(
        {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "id": task_id,
                "status": {"state": "completed"},
                "artifacts": [
                    {
                        "name": "answer",
                        "parts": [{"kind": "text", "text": answer}],
                    }
                ],
            },
        }
    )


def _extract_a2a_text(params: dict[str, Any]) -> str:
    if "question" in params:
        return str(params["question"])

    message = params.get("message", {})
    parts = message.get("parts", [])
    texts = []
    for part in parts:
        text = part.get("text") or part.get("content")
        if text:
            texts.append(str(text))
    return "\n".join(texts).strip()


def _jsonrpc_error(request_id: Any, code: int, message: str) -> JSONResponse:
    return JSONResponse(
        {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}},
        status_code=400,
    )


def main() -> None:
    import uvicorn

    uvicorn.run("app.server:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
