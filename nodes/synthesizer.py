"""
nodes/synthesizer.py — Synthesizer node for the Research Assistant Agent.

Responsibility
--------------
1. Merge Tavily results and vector-store documents into a single coherent
   context string (deduplicating by URL).
2. Extract and de-duplicate source URLs.
3. Use Gemini to evaluate whether the accumulated context is sufficient to
   answer the topic, or whether a re-planning cycle is needed.

The needs_more_research flag drives a conditional edge in the graph:
  - True  → route back to Planner (up to MAX_REPLANS times)
  - False → route forward to Responder

Design decision: asking the LLM to self-evaluate sufficiency is more robust
than a hard rule (e.g., "at least N results"). The LLM can judge topical
relevance, not just quantity.
"""

import json
import re

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from state import ResearchState

MAX_REPLANS = 2  # hard cap: even if the LLM says "more research needed",
                 # stop after this many extra planning cycles to avoid loops.


def _get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        temperature=0,
    )


_EVAL_SYSTEM = """You are evaluating whether a set of research results is \
sufficient to answer a research topic comprehensively.

Respond ONLY with a JSON object:
{"sufficient": true}  — if the context can support a thorough answer
{"sufficient": false} — if critical information is clearly missing

No preamble, no explanation, no markdown fences."""


def _build_context_string(search_results: list[dict], retrieved_docs: list[dict]) -> tuple[str, list[str]]:
    """
    Merge and deduplicate all research material into a single context string.

    Returns
    -------
    (context_str, source_urls)
    """
    seen_urls: set[str] = set()
    sources: list[str] = []
    chunks: list[str] = []

    all_items = search_results + retrieved_docs

    for item in all_items:
        url = item.get("url", "unknown")
        content = item.get("content", "").strip()

        if not content:
            continue

        if url not in seen_urls:
            seen_urls.add(url)
            sources.append(url)

        # Truncate very long content to keep context window manageable
        chunks.append(f"[Source: {url}]\n{content[:600]}")

    return "\n\n---\n\n".join(chunks), sources


def synthesizer_node(state: ResearchState) -> dict:
    """
    Synthesizer node — merges results and determines if more research is needed.

    Reads:  state["search_results"], state["retrieved_docs"],
            state["topic"], state["turn_count"]
    Writes: state["synthesis"], state["sources"], state["needs_more_research"]
    """
    search_results = state.get("search_results", [])
    retrieved_docs = state.get("retrieved_docs", [])
    topic = state.get("topic", "")
    turn_count = state.get("turn_count", 0)

    print(f"\n[Synthesizer] Merging {len(search_results)} web results + "
          f"{len(retrieved_docs)} local docs...")

    context_str, sources = _build_context_string(search_results, retrieved_docs)

    if not context_str:
        print("[Synthesizer] No usable content found — flagging for error handler.")
        return {
            "synthesis": "",
            "sources": [],
            "needs_more_research": False,  # don't loop; let error handler respond
            "error_message": "Synthesizer found no usable content from any source.",
        }

    print(f"[Synthesizer] Built context from {len(sources)} unique source(s).")

    # ── Sufficiency check ────────────────────────────────────────────────────
    # Skip the LLM check if we've already re-planned MAX_REPLANS times.
    if turn_count > MAX_REPLANS:
        print(f"[Synthesizer] Replan cap reached ({MAX_REPLANS}). Proceeding to Responder.")
        needs_more = False
    else:
        llm = _get_llm()
        eval_prompt = (
            f"Research topic: {topic}\n\n"
            f"Accumulated context ({len(sources)} sources):\n"
            f"{context_str[:2000]}..."  # truncate for the eval call
        )

        try:
            resp = llm.invoke([
                SystemMessage(content=_EVAL_SYSTEM),
                HumanMessage(content=eval_prompt),
            ])
            raw = resp.content
            if isinstance(raw, list):
                raw = raw[0].get("text", "") if isinstance(raw[0], dict) else str(raw[0])
            raw = str(raw).strip()
            raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("```").strip()
            parsed = json.loads(raw)
            needs_more = not parsed.get("sufficient", True)
            print(f"[Synthesizer] Sufficiency check -> sufficient={parsed.get('sufficient')}")
        except Exception as exc:
            print(f"[Synthesizer] Sufficiency check failed ({exc}), assuming sufficient.")
            needs_more = False

    return {
        "synthesis": context_str,
        "sources": sources,
        "needs_more_research": needs_more,
        "error_message": "",
    }
