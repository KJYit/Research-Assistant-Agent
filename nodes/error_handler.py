"""
nodes/error_handler.py — Error handler node for the Research Assistant Agent.

Responsibility
--------------
Intercept unrecoverable states (e.g., repeated empty Tavily results, synthesis
failure) and populate state["synthesis"] with a minimal degraded context so
the Responder can still produce a meaningful failure message instead of
crashing or looping forever.

This node deliberately does NOT raise an exception — the philosophy is that
the agent should always return *something* to the user, even if it's an
honest explanation of what went wrong.
"""

from state import ResearchState


def error_handler_node(state: ResearchState) -> dict:
    """
    Error handler node — logs the error and prepares graceful degradation.

    Reads:  state["error_message"], state["topic"], state["retry_count"]
    Writes: state["synthesis"] (sets to empty to trigger degraded responder path),
            state["error_message"] (formatted for display)
    """
    error_message = state.get("error_message", "Unknown error")
    topic = state.get("topic", "unknown topic")
    retry_count = state.get("retry_count", 0)

    print(f"\n[ErrorHandler] Handling error after {retry_count} retry attempt(s):")
    print(f"  Error: {error_message}")

    formatted_error = (
        f"Tool failure after {retry_count} attempt(s): {error_message}. "
        f"Topic: '{topic}'."
    )

    return {
        "synthesis": "",          # empty synthesis triggers degraded path in responder
        "sources": [],
        "error_message": formatted_error,
        "needs_more_research": False,
    }
