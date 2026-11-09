"""Separate checkpointed session state from private research-turn state."""

import operator
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import BaseMessage

from tools.research_providers import Candidate, Judgment


class SessionState(TypedDict, total=False):
    topic: str
    messages: list[BaseMessage]
    final_answer: str
    sources: list[str]


class TurnState(TypedDict, total=False):
    topic: str
    history: list[BaseMessage]
    standalone_question: str
    original_queries: dict[int, str]
    queries: list[str]
    searched_queries: list[str]
    retries: int
    local_rankings: list[list[Candidate]]
    web_rankings: list[list[Candidate]]
    fallbacks: Annotated[list[str], operator.add]
    fused_candidates: list[Candidate]
    selected_candidates: list[Candidate]
    judgments: list[Judgment]
    rerank_available: bool
    coverage: Literal["sufficient", "incomplete", "uncertain", "insufficient"]
    missing_query_ids: list[int]
    final_answer: str
    sources: list[str]
