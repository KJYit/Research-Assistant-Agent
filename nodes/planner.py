"""
nodes/planner.py — Planner node for the Research Assistant Agent.

Responsibility
--------------
Given the user's topic (and any prior conversation context), produce a list
of 2–4 targeted sub-questions. Each sub-question will drive one Tavily
search call in the Researcher node.

Design decision: the Planner uses a structured JSON prompt rather than
free-form output so the downstream parser is deterministic. Asking Gemini
to return raw JSON inside a code fence is more reliable than hoping it
follows a schema with prose output.
"""

import json
import re

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from state import ResearchState


# ── LLM initialisation ───────────────────────────────────────────────────────
# Temperature 0 for the Planner: we want consistent, structured decomposition,
# not creative variation. The Responder uses a slightly higher temperature.
def _get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        temperature=0,
    )


_PLANNER_SYSTEM = """You are an expert research planner. Your ONLY job is to \
decompose a research topic into 2-4 specific, searchable sub-questions.

Rules:
- Output ONLY a JSON object with a single key "sub_questions" containing a list of strings.
- No preamble, no explanation, no markdown fences.
- Each sub-question must be self-contained and searchable on a search engine.
- Avoid redundant sub-questions.
- If this is a follow-up turn, refine or deepen the sub-questions based on \
  what was already researched (provided in context).

Example output:
{"sub_questions": ["What is X?", "How does Y affect Z?", "What are the latest trends in W?"]}
"""


def planner_node(state: ResearchState) -> dict:
    """
    Planner node — decomposes the topic into targeted sub-questions.

    Reads:  state["topic"], state["messages"], state["turn_count"]
    Writes: state["sub_questions"], state["turn_count"]
    """
    llm = _get_llm()
    topic = state.get("topic", "")
    turn_count = state.get("turn_count", 0)

    # Build context string from prior conversation for multi-turn awareness
    history_context = ""
    prior_answers = [
        m for m in state.get("messages", [])
        if isinstance(m, AIMessage)
    ]
    if prior_answers:
        last = prior_answers[-1].content
        history_context = f"\n\nPreviously researched context (summarised):\n{last[:800]}"

    user_prompt = (
        f"Research topic: {topic}"
        f"{history_context}"
        f"\n\nThis is research turn #{turn_count + 1}. "
        "Generate focused sub-questions."
    )

    messages = [
        SystemMessage(content=_PLANNER_SYSTEM),
        HumanMessage(content=user_prompt),
    ]

    print(f"\n[Planner] Decomposing topic: '{topic}' (turn {turn_count + 1})")

    try:
        response = llm.invoke(messages)
        raw = response.content
        if isinstance(raw, list):
            raw = raw[0].get("text", "") if isinstance(raw[0], dict) else str(raw[0])
        raw = str(raw).strip()

        # Strip accidental markdown fences if Gemini adds them
        raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("```").strip()

        parsed = json.loads(raw)
        sub_questions = parsed.get("sub_questions", [])

        if not sub_questions:
            raise ValueError("Empty sub_questions list returned.")

        print(f"[Planner] Generated {len(sub_questions)} sub-questions:")
        for i, q in enumerate(sub_questions, 1):
            print(f"  {i}. {q}")

        return {
            "sub_questions": sub_questions,
            "turn_count": turn_count + 1,
            "error_message": "",  # clear any prior error
        }

    except Exception as exc:
        # Fallback: treat the raw topic as a single sub-question so the
        # graph can still make progress rather than halting entirely.
        print(f"[Planner] ERROR: {exc} — falling back to raw topic.")
        return {
            "sub_questions": [topic],
            "turn_count": turn_count + 1,
            "error_message": f"Planner fallback triggered: {exc}",
        }
