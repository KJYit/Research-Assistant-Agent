"""
graph.py — LangGraph StateGraph definition for the Research Assistant Agent.

Graph topology
--------------

    [START]
       │
       ▼
   [planner]  ◄──────────────────────────────────────────────┐
       │                                                      │
       ▼                                                      │
  [researcher]                                               │
       │                                                      │
       ├─── retry_count > 1 ──► [error_handler] ─► [responder] ─► [END]
       │                                                      │
       ▼                                                      │
  [retriever]                                                │
       │                                                      │
       ▼                                                      │
  [synthesizer]                                              │
       │                                                      │
       ├─── needs_more_research=True ─────────────────────────┘
       │
       ▼
  [responder]
       │
       ▼
    [END]

Conditional edges
-----------------
1. after_researcher : checks retry_count → routes to error_handler or retriever
2. after_synthesizer: checks needs_more_research → routes to planner or responder
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from state import ResearchState
from nodes import (
    planner_node,
    researcher_node,
    retriever_node,
    synthesizer_node,
    responder_node,
    error_handler_node,
)


# ── Conditional edge functions ───────────────────────────────────────────────

def route_after_researcher(state: ResearchState) -> str:
    """
    Route based on whether Tavily returned any results.

    Threshold: if retry_count > 1, the search has failed twice — send to
    error handler. Otherwise, continue to retriever.
    """
    retry_count = state.get("retry_count", 0)
    if retry_count > 1:
        print(f"[Router] retry_count={retry_count} > 1 -> routing to error_handler")
        return "error_handler"
    print(f"[Router] retry_count={retry_count} -> routing to retriever")
    return "retriever"


def route_after_synthesizer(state: ResearchState) -> str:
    """
    Route based on the Synthesizer's sufficiency judgment.

    If needs_more_research is True AND we haven't exceeded the replan cap,
    loop back to the Planner. Otherwise, proceed to the Responder.
    """
    needs_more = state.get("needs_more_research", False)
    turn_count = state.get("turn_count", 0)

    # turn_count is already incremented by the Planner, so cap at 3 total turns
    if needs_more and turn_count < 3:
        print(f"[Router] needs_more_research=True, turn={turn_count} -> re-planning")
        return "planner"

    print(f"[Router] needs_more_research={needs_more}, turn={turn_count} -> responder")
    return "responder"


# ── Graph construction ───────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """
    Construct and compile the LangGraph StateGraph.

    MemorySaver is used as the checkpointer, enabling thread-level
    multi-turn persistence. Each conversation thread is identified by a
    thread_id passed at invocation time.
    """
    builder = StateGraph(ResearchState)

    # Register nodes
    builder.add_node("planner",       planner_node)
    builder.add_node("researcher",    researcher_node)
    builder.add_node("retriever",     retriever_node)
    builder.add_node("synthesizer",   synthesizer_node)
    builder.add_node("responder",     responder_node)
    builder.add_node("error_handler", error_handler_node)

    # Unconditional edges
    builder.add_edge(START,          "planner")
    builder.add_edge("planner",      "researcher")
    builder.add_edge("retriever",    "synthesizer")
    builder.add_edge("error_handler","responder")
    builder.add_edge("responder",    END)

    # Conditional edge 1: after researcher -> retriever OR error_handler
    builder.add_conditional_edges(
        "researcher",
        route_after_researcher,
        {
            "retriever":     "retriever",
            "error_handler": "error_handler",
        },
    )

    # Conditional edge 2: after synthesizer -> responder OR planner (re-plan)
    builder.add_conditional_edges(
        "synthesizer",
        route_after_synthesizer,
        {
            "responder": "responder",
            "planner":   "planner",
        },
    )

    # Compile with in-memory checkpointer for multi-turn session persistence
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)
    return graph


# Module-level compiled graph instance (imported by main.py and cli.py)
research_graph = build_graph()