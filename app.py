import json
import sys
from pathlib import Path

from agent.graph import build_research_graph, build_qa_graph
from sessions.manager import (
    create,
    list_sessions,
    get,
    set_paper,
    add_message,
)

research_graph = build_research_graph()
qa_graph = build_qa_graph()

def print_paper(p, number=None):
    prefix = f"{number}. " if number is not None else ""
    print(f"{prefix}{p['title']}")
    print(f"   arXiv: {p['arxiv_id']}")
    print(f"   Date : {p['published']}")
    print(f"   URL  : {p['abs_url']}")
    print()

def choose_candidate(candidates):
    print("\nCandidate papers:\n")
    for i, paper in enumerate(candidates, 1):
        print_paper(paper, i)

    while True:
        raw = input("Select paper number (or q): ").strip()
        if raw.lower() == "q":
            return None
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(candidates):
                return candidates[idx]
        except ValueError:
            pass
        print("Invalid selection.")

def print_briefing(briefing):
    print("\n" + "=" * 72)
    print("EXECUTIVE BRIEFING")
    print("=" * 72)
    print(f"\n{briefing['title']}")
    print(f"arXiv: {briefing['arxiv_id']}")
    print(f"Authors: {', '.join(briefing['authors'])}")
    print(f"Published: {briefing['publish_date']}")
    print(f"Link: {briefing['link']}")

    print("\nWHY THIS PAPER MATTERS")
    print(briefing["why_it_matters"])

    print("\nPROBLEM")
    print(briefing["problem_statement"])

    print("\nMETHOD")
    for x in briefing["method"]:
        print(f"  • {x}")

    print("\nKEY RESULTS / CLAIMS")
    for x in briefing["key_results"]:
        print(f"  • {x}")

    print("\nLIMITATIONS")
    for x in briefing["limitations"]:
        print(f"  • {x}")

    print("\nFOLLOW-UP QUESTIONS")
    for x in briefing["follow_up_questions"]:
        print(f"  • {x}")

    print("\n" + "=" * 72)

def select_or_create_session():
    sessions = list_sessions()
    if not sessions:
        sid = create()
        print(f"Created session: {sid}")
        return sid

    print("\nSessions:")
    for s in sessions[-10:]:
        paper = s.get("paper")
        paper_name = paper["title"] if paper else "No paper selected"
        print(f"  {s['id']} | {s['title']} | {paper_name}")

    raw = input(
        "\nEnter session ID to continue, or press Enter for a new session: "
    ).strip()

    if raw and get(raw):
        return raw

    sid = create()
    print(f"Created session: {sid}")
    return sid

def search_and_open(sid, user_input):
    result = research_graph.invoke({
        "user_input": user_input,
        "session_id": sid,
    })

    if result.get("error"):
        print(f"\nError: {result['error']}")
        return None, None

    candidates = result.get("candidates", [])
    if result.get("input_type") == "topic" and len(candidates) > 1:
        selected = choose_candidate(candidates)
        if not selected:
            return None, None

        # Run the indexing/briefing portion again for the selected paper.
        result = research_graph.invoke({
            "user_input": selected["arxiv_id"],
            "session_id": sid,
        })
    else:
        selected = result["selected_paper"]

    set_paper(sid, selected)
    return selected, result

def qa_loop(sid, paper):
    print(
        "\nQA mode. Commands: /new, /paper, /sessions, /exit"
    )

    while True:
        question = input("\nYou: ").strip()

        if not question:
            continue

        if question in ("/exit", "exit"):
            return "exit"

        if question == "/new":
            return "new"

        if question == "/paper":
            print_paper(paper)
            continue

        if question == "/sessions":
            for s in list_sessions():
                print(f"{s['id']} | {s['title']}")
            continue

        add_message(sid, "user", question)

        try:
            result = qa_graph.invoke({
                "question": question,
                "selected_paper": paper,
                "session_id": sid,
            })
            answer = result["answer"]
            sources = result.get("sources", [])

            print(f"\nAssistant:\n{answer}")

            unique = sorted({
                (s["page"], s["type"])
                for s in sources
            })
            if unique:
                print(
                    "Sources: "
                    + ", ".join(
                        f"page {p} ({t})"
                        for p, t in unique
                    )
                )

            add_message(
                sid,
                "assistant",
                answer,
                sources,
            )
        except Exception as exc:
            print(f"\nQA error: {exc}")

def main():
    print("""
============================================================
              AUTONOMOUS arXiv PAPER AGENT
============================================================

Enter:
  • a research topic
  • an arXiv ID
  • an arXiv URL

The agent searches/fetches the paper, caches its local
PDF and Chroma embeddings, creates a structured briefing,
then opens grounded QA mode.

Commands:
  /new       Start another session
  /paper     Show current paper
  /sessions  List sessions
  /exit      Exit

============================================================
""")

    sid = select_or_create_session()

    while True:
        user_input = input(
            "\nResearch topic or arXiv ID/URL (or /exit): "
        ).strip()

        if user_input in ("/exit", "exit"):
            break
        if not user_input:
            continue

        try:
            paper, result = search_and_open(sid, user_input)
            if not paper:
                continue

            if result.get("briefing"):
                print_briefing(result["briefing"])

            action = qa_loop(sid, paper)

            if action == "exit":
                break
            if action == "new":
                sid = create()
                print(f"\nNew session: {sid}")

        except Exception as exc:
            print(f"\nAgent error: {exc}")
            print(
                "The local cache is preserved, so you can retry "
                "without necessarily re-downloading the paper."
            )

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        sys.exit(0)
