"""
main.py — Entry point for the Research Assistant Agent.

Usage
-----
    python main.py

Starts an interactive multi-turn CLI session. Each conversation is tied to a
thread_id so LangGraph's MemorySaver can restore state between queries.

Multi-turn design
-----------------
The same thread_id is reused across all turns in a session. LangGraph
persists the full ResearchState (including messages) in the MemorySaver
checkpoint store, so each new query picks up exactly where the last left off.

To start a new independent research session, type 'new' at the prompt or
restart the script (a new UUID thread_id is generated per script run).
"""

import os
import sys
import uuid
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

from graph import research_graph

# ── Environment variable validation ─────────────────────────────────────────
def _check_env():
    missing = []
    if not os.getenv("GOOGLE_API_KEY"):
        missing.append("GOOGLE_API_KEY")
    if not os.getenv("TAVILY_API_KEY"):
        missing.append("TAVILY_API_KEY")

    if missing:
        print("\n⚠️  Missing required environment variables:")
        for var in missing:
            print(f"   • {var}")
        print("\nSet them before running:")
        print("   export GOOGLE_API_KEY='your-key-here'")
        print("   export TAVILY_API_KEY='your-key-here'")
        print("\nOr create a .env file and use: pip install python-dotenv")
        sys.exit(1)


def run_research_turn(
    topic: str,
    thread_id: str,
    verbose: bool = True,
) -> str:
    """
    Execute one research turn and return the final answer.

    Parameters
    ----------
    topic     : the user's research question or topic
    thread_id : LangGraph thread identifier for session persistence
    verbose   : whether to stream intermediate node output

    Returns
    -------
    The final answer string.
    """
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "messages":           [HumanMessage(content=topic)],
        "topic":              topic,
        "sub_questions":      [],
        "search_results":     [],
        "retrieved_docs":     [],
        "sources":            [],
        "synthesis":          "",
        "final_answer":       "",
        "turn_count":         0,
        "retry_count":        0,
        "error_message":      "",
        "needs_more_research": False,
    }

    print(f"\n{'='*60}")
    print(f"  Research topic: {topic}")
    print(f"  Thread ID:      {thread_id}")
    print(f"{'='*60}")

    # invoke() runs the full graph synchronously and returns the final state
    final_state = research_graph.invoke(initial_state, config=config)

    answer = final_state.get("final_answer", "No answer generated.")
    sources = final_state.get("sources", [])

    return answer, sources


def main():
    """Interactive multi-turn research session."""
    _check_env()

    # Generate a unique thread ID for this session
    thread_id = str(uuid.uuid4())

    print("\n" + "="*60)
    print("   🔬 Research Assistant Agent (powered by Gemini + Tavily)")
    print("="*60)
    print("Commands:")
    print("  • Type a research topic and press Enter")
    print("  • Type 'new'  to start a fresh research session")
    print("  • Type 'quit' or Ctrl+C to exit")
    print("="*60 + "\n")

    while True:
        try:
            user_input = input("Research topic > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() == "quit":
            print("Goodbye!")
            break

        if user_input.lower() == "new":
            thread_id = str(uuid.uuid4())
            print(f"\n✓ Started new session (thread: {thread_id[:8]}...)\n")
            continue

        try:
            answer, sources = run_research_turn(
                topic=user_input,
                thread_id=thread_id,
            )

            print("\n" + "─"*60)
            print("ANSWER")
            print("─"*60)
            print(answer)
            print("─"*60 + "\n")

        except Exception as exc:
            print(f"\n❌ Unexpected error: {exc}")
            print("The agent encountered an unhandled exception. Please try again.\n")


if __name__ == "__main__":
    main()
