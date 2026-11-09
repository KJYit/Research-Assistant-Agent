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
checkpoints conversation history and the latest public result in memory.

To start a new independent research session, type 'new' at the prompt or
restart the script (a new UUID thread_id is generated per script run).
"""

import os
import uuid
from dotenv import load_dotenv

load_dotenv()

from graph import invoke_research_turn

# ── Environment variable validation ─────────────────────────────────────────
def _check_env():
    if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
        print("Google credentials are missing. Query rewriting, semantic search, "
              "and cited answer generation may be unavailable.")


def run_research_turn(
    topic: str,
    thread_id: str,
    verbose: bool = True,
) -> tuple[str, list[str]]:
    """
    Execute one research turn and return the final answer.

    Parameters
    ----------
    topic     : the user's research question or topic
    thread_id : LangGraph thread identifier for session persistence
    verbose   : whether to display the research topic and thread ID

    Returns
    -------
    The final answer and its source URLs.
    """
    if verbose:
        print(f"\n{'='*60}\n  Research topic: {topic}\n  Thread ID:      {thread_id}")
        print(f"{'='*60}\n")

    final_state = invoke_research_turn(topic, thread_id)

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
            if sources:
                print("\nSources")
                for index, url in enumerate(sources, 1):
                    print(f"[{index}] {url}")
            print("─"*60 + "\n")

        except Exception as exc:
            print(f"\n❌ Unexpected error: {exc}")
            print("The agent encountered an unhandled exception. Please try again.\n")


if __name__ == "__main__":
    main()
