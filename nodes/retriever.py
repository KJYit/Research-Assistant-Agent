"""
nodes/retriever.py — Retriever node for the Research Assistant Agent.

Responsibility
--------------
Augment the Tavily web results with documents from the in-memory knowledge
base (Tool 2). Retrieval runs against the original topic and each sub-question
to maximise recall across the local corpus.

Why a second tool?
------------------
The two-tool design mirrors a real RAG architecture:
  - Tavily  → live, broad, web-scale knowledge
  - Vector store → curated, stable, domain-specific knowledge

Together they improve both coverage and reliability (the vector store always
returns results even when Tavily is rate-limited or the topic is niche).
"""

from tools.search_tools import run_vector_retrieve
from state import ResearchState


def retriever_node(state: ResearchState) -> dict:
    """
    Retriever node — queries the in-memory TF-IDF vector store.

    Reads:  state["topic"], state["sub_questions"]
    Writes: state["retrieved_docs"]
    """
    topic = state.get("topic", "")
    sub_questions = state.get("sub_questions", [])

    # Build a deduplicated set of queries: topic + all sub-questions
    queries = list(dict.fromkeys([topic] + sub_questions))

    print(f"\n[Retriever] Querying vector store with {len(queries)} query/queries...")

    all_docs: list[dict] = []
    seen_urls: set[str] = set()

    for query in queries:
        docs = run_vector_retrieve(query, top_k=2)
        for doc in docs:
            # Deduplicate by URL so the same document isn't included twice
            if doc["url"] not in seen_urls:
                seen_urls.add(doc["url"])
                all_docs.append(doc)

    print(f"[Retriever] Retrieved {len(all_docs)} unique document(s) from vector store.")
    for doc in all_docs:
        print(f"  -> {doc['title']} ({doc['url']})")

    return {
        "retrieved_docs": all_docs,
    }
