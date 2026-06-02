import argparse
import sys

from app.config import ConfigError, configure_stdio, get_settings
from app.memory import append_session_turn


def chat(session_id: str) -> int:
    from langchain_core.messages import HumanMessage

    from app.graph import graph

    settings = get_settings()
    messages = []
    print("LangGraph agent is ready. Type 'exit' to quit.")

    while True:
        question = input("\nYou: ").strip()
        if question.lower() in {"exit", "quit", "q"}:
            return 0
        if not question:
            continue

        messages.append(HumanMessage(content=question))
        try:
            result = graph.invoke(
                {"messages": messages},
                config={
                    "recursion_limit": settings.recursion_limit,
                    "configurable": {"session_id": session_id},
                },
            )
        except Exception as exc:
            print(f"Agent error: {exc}")
            continue

        messages = result["messages"]
        answer = messages[-1].content
        append_session_turn(question, answer, session_id=session_id)
        print(f"Agent: {answer}")


def main(argv: list[str] | None = None) -> int:
    configure_stdio()
    parser = argparse.ArgumentParser(description="Run the minimal LangGraph agent.")
    parser.add_argument("question", nargs="*", help="Question or task for one-shot mode.")
    parser.add_argument("--chat", action="store_true", help="Start interactive chat mode.")
    parser.add_argument("--plan", action="store_true", help="Create a coding plan without editing.")
    parser.add_argument("--rag", action="store_true", help="Use the explicit agentic RAG graph.")
    parser.add_argument("--general", action="store_true", help="Force the general tool agent.")
    parser.add_argument("--trace", action="store_true", help="Print detailed general-agent trace.")
    parser.add_argument("--session", default="default", help="Conversation memory session id.")
    args = parser.parse_args(argv)

    try:
        if args.chat:
            return chat(args.session)

        question = " ".join(args.question).strip()
        if not question:
            question = "现在北京时间几点？请先用工具获取时间，再用一句话回答。"
        if args.plan:
            from app.runner import answer_question

            print(answer_question(question, mode="plan", session_id=args.session))
            return 0
        if args.trace:
            from app.graph import run_with_trace
            from app.memory import append_session_turn

            answer, trace = run_with_trace(question, session_id=args.session)
            append_session_turn(question, answer, session_id=args.session)
            print(trace)
            return 0
        if args.rag:
            from app.runner import answer_question

            print(answer_question(question, mode="rag", session_id=args.session))
            return 0
        if args.general:
            from app.runner import answer_question

            print(answer_question(question, mode="general", session_id=args.session))
            return 0
        from app.runner import answer_question

        print(answer_question(question, session_id=args.session))
        return 0
    except ConfigError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Agent error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
