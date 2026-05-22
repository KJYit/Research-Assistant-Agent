"""
state.py — Defines the shared state schema for the Research Assistant Agent.

All nodes read from and write to this TypedDict. LangGraph passes the full
state object between nodes, so every piece of information the agent needs
across its lifetime lives here.
"""

from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
import operator


class ResearchState(TypedDict):
    """
    Centralised state object that flows through every node in the graph.

    Fields
    ------
    messages : list[BaseMessage]
        Full conversation history (HumanMessage / AIMessage). Used for
        multi-turn memory — LangGraph's add_messages reducer appends rather
        than replaces, preserving turn history automatically.

    topic : str
        The high-level research topic extracted from the latest user message.

    sub_questions : list[str]
        Decomposed sub-questions produced by the Planner node. Each sub-
        question drives one Tavily search call in the Researcher node.

    search_results : list[dict]
        Raw results returned by Tavily. Each dict contains 'query', 'content',
        and 'url' keys. Accumulated across all sub-questions.

    retrieved_docs : list[dict]
        Results from the in-memory vector store (second tool). Same shape as
        search_results for uniform downstream handling.

    sources : list[str]
        De-duplicated list of URLs / source labels assembled by the
        Synthesizer node. Injected into the final response.

    synthesis : str
        Merged, cleaned context string produced by the Synthesizer node.
        Passed to the Responder to draft the final answer.

    final_answer : str
        The polished, citation-aware answer returned to the user.

    turn_count : int
        Number of completed research turns in this session. Used to prevent
        infinite re-planning loops (hard cap at 3 re-plan cycles).

    retry_count : int
        Tracks consecutive empty-search retries so the error handler can
        intervene after 1 failed attempt.

    error_message : str
        Populated by the error handler node; empty string when no error.

    needs_more_research : bool
        Signal from the Synthesizer to the router: True means loop back to
        the Planner with refined sub-questions; False means proceed to the
        Responder.
    """

    # ── Conversation memory ──────────────────────────────────────────────────
    # Annotated with operator.add so LangGraph *appends* new messages instead
    # of overwriting the list on each node update.
    messages: Annotated[Sequence[BaseMessage], operator.add]

    # ── Research context ─────────────────────────────────────────────────────
    topic: str
    sub_questions: list[str]
    search_results: list[dict]
    retrieved_docs: list[dict]
    sources: list[str]
    synthesis: str
    final_answer: str

    # ── Control flow ─────────────────────────────────────────────────────────
    turn_count: int
    retry_count: int
    error_message: str
    needs_more_research: bool
