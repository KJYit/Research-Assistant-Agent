"""
nodes/responder.py — Responder node for the Research Assistant Agent.

Responsibility
--------------
Given the synthesized context and de-duplicated sources, produce a
well-structured, cited research answer and append it to the conversation
history so future turns have full context.

Design decisions
----------------
- Temperature 0.3: low enough for factual accuracy, high enough to produce
  natural prose rather than robotic bullet dumps.
- Citations are inline (e.g. "[1]") with a numbered source list at the end,
  matching academic convention.
- The full AIMessage is appended to state["messages"] to support multi-turn
  memory — subsequent Planner calls include this answer as prior context.
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from state import ResearchState


def _get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        temperature=0.3,
    )


_RESPONDER_SYSTEM = """You are an expert research analyst. Your job is to write \
a comprehensive, well-structured answer to a research topic using ONLY the \
provided context. 

Requirements:
1. Write in clear, flowing paragraphs (not just bullet points).
2. Cite sources inline using [1], [2], etc. format.
3. End with a numbered "Sources" section listing each URL.
4. If the context is insufficient for some aspects, say so honestly rather \
   than hallucinating.
5. Aim for 200–400 words — thorough but concise.
"""


def responder_node(state: ResearchState) -> dict:
    """
    Responder node — generates the final cited answer.

    Reads:  state["topic"], state["synthesis"], state["sources"],
            state["messages"], state["error_message"]
    Writes: state["final_answer"], state["messages"] (appends AIMessage)
    """
    topic = state.get("topic", "")
    synthesis = state.get("synthesis", "")
    sources = state.get("sources", [])
    error_message = state.get("error_message", "")

    # ── Handle degraded state (error handler set a message) ─────────────────
    if error_message and not synthesis:
        degraded_answer = (
            f"I was unable to retrieve sufficient information about '{topic}' "
            f"due to the following issue:\n\n> {error_message}\n\n"
            "Please try:\n"
            "- Rephrasing the topic more specifically\n"
            "- Checking your TAVILY_API_KEY environment variable\n"
            "- Trying again in a few moments\n\n"
            "No sources are available for this response."
        )
        return {
            "final_answer": degraded_answer,
            "messages": [AIMessage(content=degraded_answer)],
        }

    # ── Build numbered source list ───────────────────────────────────────────
    source_list = "\n".join(f"[{i+1}] {url}" for i, url in enumerate(sources))
    source_reference = (
        f"\n\nThe following {len(sources)} source(s) are available:\n{source_list}"
        if sources else "\n\nNo external sources available."
    )

    user_prompt = (
        f"Research topic: {topic}\n\n"
        f"Context gathered from research:\n{synthesis}\n"
        f"{source_reference}\n\n"
        "Write a comprehensive, cited research answer."
    )

    llm = _get_llm()

    print(f"\n[Responder] Generating final answer for topic: '{topic}'")

    try:
        response = llm.invoke([
            SystemMessage(content=_RESPONDER_SYSTEM),
            HumanMessage(content=user_prompt),
        ])
        raw = response.content
        if isinstance(raw, list):
            raw = raw[0].get("text", "") if isinstance(raw[0], dict) else str(raw[0])
        answer = str(raw).strip()

    except Exception as exc:
        answer = (
            f"I encountered an error while generating the response: {exc}\n\n"
            "The research was completed but the answer could not be formatted. "
            "Please try again."
        )

    print("[Responder] Answer generated successfully.")

    return {
        "final_answer": answer,
        # Append to messages so the next turn has full conversation context
        "messages": [AIMessage(content=answer)],
    }
