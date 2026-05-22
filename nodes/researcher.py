"""
nodes/researcher.py — Researcher node for the Research Assistant Agent.

Responsibility
--------------
For each sub-question produced by the Planner, execute one Tavily web
search. Accumulate all results and track the retry count for empty responses.

Design decision: sub-questions are searched sequentially rather than in
parallel. This keeps the implementation simple and avoids hitting Tavily's
concurrent request limits on the free tier. For a production system,
asyncio.gather() would be the right approach.

Error handling strategy
-----------------------
If Tavily returns empty results for ALL sub-questions, the retry_count is
incremented. The router (in graph.py) checks this value and routes to the
error_handler node after 1 failed attempt, which substitutes a graceful
degradation message rather than letting the agent loop forever.
"""

from tools.search_tools import run_tavily_search
from state import ResearchState


def researcher_node(state: ResearchState) -> dict:
    """
    Researcher node — calls Tavily for each sub-question.

    Reads:  state["sub_questions"], state["retry_count"]
    Writes: state["search_results"], state["retry_count"], state["error_message"]
    """
    sub_questions = state.get("sub_questions", [])
    retry_count = state.get("retry_count", 0)

    print(f"\n[Researcher] Running {len(sub_questions)} Tavily search(es)...")

    all_results: list[dict] = []

    for question in sub_questions:
        print(f"  -> Searching: '{question}'")
        results = run_tavily_search(question, max_results=4)

        if results:
            print(f"    [+] Got {len(results)} result(s).")
        else:
            print(f"    [-] No results returned for this query.")

        all_results.extend(results)

    # Determine retry state
    if not all_results:
        new_retry_count = retry_count + 1
        print(f"[Researcher] All searches returned empty. Retry count: {new_retry_count}")
        return {
            "search_results": [],
            "retry_count": new_retry_count,
            "error_message": "Tavily returned no results for all sub-questions.",
        }

    print(f"[Researcher] Total results collected: {len(all_results)}")
    return {
        "search_results": all_results,
        "retry_count": 0,          # reset on success
        "error_message": "",
    }
