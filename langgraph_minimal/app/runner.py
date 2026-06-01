from collections.abc import Iterator


def answer_question(
    question: str,
    mode: str = "auto",
    session_id: str = "default",
    persist: bool = True,
) -> str:
    if mode == "rag":
        from app.rag_graph import answer_with_rag

        answer = answer_with_rag(question, session_id=session_id)
    elif mode == "general":
        from app.graph import run_once

        answer = run_once(question, session_id=session_id)
    else:
        from app.router_graph import answer as route_answer

        answer = route_answer(question, session_id=session_id)

    if persist:
        from app.memory import append_session_turn

        append_session_turn(question, answer, session_id=session_id)
    return answer


def answer_events(
    question: str,
    mode: str = "auto",
    session_id: str = "default",
) -> Iterator[dict]:
    yield {"event": "received", "question": question, "mode": mode, "session_id": session_id}
    route = mode
    if mode == "auto":
        from app.router_graph import router

        route = router(
            {"question": question, "session_id": session_id, "route": "", "answer": ""}
        )["route"]
    yield {"event": "route", "route": route}
    yield {"event": "running", "route": route}
    answer = answer_question(question, mode=route, session_id=session_id)
    yield {"event": "final", "route": route, "answer": answer}
