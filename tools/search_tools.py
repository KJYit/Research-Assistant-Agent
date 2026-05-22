"""
tools/search_tools.py — Tool definitions for the Research Assistant Agent.

Two tools are registered:
  1. tavily_search   — live web search via Tavily API (LangChain-native)
  2. vector_retrieve — keyword-ranked retrieval from an in-memory document store

Both return a list[dict] with uniform keys: query, content, url.
This consistency means downstream nodes never need to branch on tool type.
"""

import os
import math
import re
from collections import Counter
from typing import Any

from langchain_community.tools.tavily_search import TavilySearchResults


# ── Tool 1: Tavily Web Search ────────────────────────────────────────────────

def get_tavily_tool(max_results: int = 4) -> TavilySearchResults:
    """
    Returns a configured TavilySearchResults tool instance.

    Tavily is preferred over DuckDuckGo for LangGraph agents because it
    returns structured JSON (title, url, content) instead of raw HTML,
    which eliminates a parsing step.

    Requires TAVILY_API_KEY in environment.
    """
    return TavilySearchResults(
        max_results=max_results,
        search_depth="advanced",       # deeper crawl for research tasks
        include_answer=True,           # include Tavily's own summary
        include_raw_content=False,     # raw HTML not needed; saves tokens
    )


def run_tavily_search(query: str, max_results: int = 4) -> list[dict]:
    """
    Execute a single Tavily search and normalise results.

    Parameters
    ----------
    query       : the search string
    max_results : max number of results to return

    Returns
    -------
    list of dicts with keys: query, content, url
    Empty list on any failure (caller handles the empty-result case).
    """
    try:
        tool = get_tavily_tool(max_results)
        raw = tool.invoke({"query": query})

        results = []
        for item in raw:
            results.append({
                "query":   query,
                "content": item.get("content", ""),
                "url":     item.get("url", "No URL"),
                "title":   item.get("title", ""),
            })
        return results

    except Exception as exc:
        # Surface the error string so the error-handler node can log it
        print(f"[Tavily] Search failed for '{query}': {exc}")
        return []


# ── Tool 2: In-Memory Vector Store (TF-IDF keyword retriever) ───────────────
# Design decision: using a hand-rolled TF-IDF retriever instead of FAISS/
# Chroma keeps the project dependency-light and avoids the need for an
# embeddings API key. For production use, swap _score_doc for a real
# similarity function.

# Seed corpus — representative research documents baked in so the tool
# works offline and provides meaningful results for common CS/AI topics.
_DOCUMENT_STORE: list[dict] = [
    {
        "id": "doc_001",
        "title": "Introduction to Large Language Models",
        "content": (
            "Large language models (LLMs) are neural networks trained on massive text corpora "
            "using self-supervised objectives. Models like GPT-4, Gemini, and Claude learn "
            "statistical patterns across billions of tokens, enabling emergent capabilities such "
            "as in-context learning, chain-of-thought reasoning, and instruction following. "
            "Scaling laws suggest that performance improves predictably with model size and data."
        ),
        "url": "internal://docs/llm_intro",
    },
    {
        "id": "doc_002",
        "title": "Retrieval-Augmented Generation (RAG)",
        "content": (
            "RAG combines a retrieval component with a generative model. At inference time, "
            "relevant documents are fetched from a knowledge base and prepended to the prompt. "
            "This reduces hallucination and allows the model to cite sources. Common retrieval "
            "backends include FAISS, Chroma, Pinecone, and Weaviate. Dense passage retrieval "
            "(DPR) and BM25 are popular scoring methods."
        ),
        "url": "internal://docs/rag_overview",
    },
    {
        "id": "doc_003",
        "title": "Agentic AI and Tool Use",
        "content": (
            "AI agents extend LLMs with the ability to call external tools — web search, "
            "code interpreters, databases — and act over multiple steps. Frameworks such as "
            "LangGraph, LangChain, AutoGPT, and CrewAI structure this as a graph of nodes "
            "with conditional routing. ReAct (Reason + Act) is the dominant prompting strategy "
            "for guiding tool selection."
        ),
        "url": "internal://docs/agentic_ai",
    },
    {
        "id": "doc_004",
        "title": "Transformer Architecture",
        "content": (
            "The Transformer, introduced in 'Attention is All You Need' (Vaswani et al., 2017), "
            "replaced recurrent networks with multi-head self-attention. The encoder-decoder "
            "structure enables sequence-to-sequence tasks. Decoder-only variants (GPT family) "
            "dominate language modelling. Key components: positional encoding, layer normalisation, "
            "feed-forward sublayers, and residual connections."
        ),
        "url": "internal://docs/transformer",
    },
    {
        "id": "doc_005",
        "title": "Prompt Engineering Best Practices",
        "content": (
            "Effective prompts are specific, provide context, and include examples (few-shot). "
            "Chain-of-thought (CoT) prompting improves multi-step reasoning by asking the model "
            "to show its work. System prompts set persona and constraints. Temperature controls "
            "randomness; lower values suit factual tasks, higher values suit creative tasks."
        ),
        "url": "internal://docs/prompt_engineering",
    },
    {
        "id": "doc_006",
        "title": "Reinforcement Learning from Human Feedback (RLHF)",
        "content": (
            "RLHF fine-tunes LLMs to align with human preferences. A reward model is trained "
            "on human comparison data, then the LLM is optimised against it using PPO. "
            "Variants include DPO (Direct Preference Optimisation) which skips the reward model. "
            "RLHF is central to the safety and helpfulness improvements in models like ChatGPT "
            "and Claude."
        ),
        "url": "internal://docs/rlhf",
    },
    {
        "id": "doc_007",
        "title": "Computer Vision and CNNs",
        "content": (
            "Convolutional Neural Networks (CNNs) extract spatial hierarchies from images. "
            "Landmark architectures: AlexNet, VGG, ResNet, EfficientNet. Vision Transformers (ViT) "
            "apply attention to image patches. Applications include object detection (YOLO, Faster R-CNN), "
            "semantic segmentation, and action recognition from video sequences."
        ),
        "url": "internal://docs/cv_cnns",
    },
    {
        "id": "doc_008",
        "title": "Graph Neural Networks",
        "content": (
            "GNNs generalise deep learning to graph-structured data. Message passing propagates "
            "features across edges. Variants: GCN, GAT, GraphSAGE. Applications include molecular "
            "property prediction, social network analysis, recommendation systems, and knowledge "
            "graph reasoning. Oversmoothing is a key limitation at large depths."
        ),
        "url": "internal://docs/gnns",
    },
    {
        "id": "doc_009",
        "title": "Cybersecurity and AI",
        "content": (
            "AI is applied in cybersecurity for intrusion detection, malware classification, "
            "vulnerability discovery, and threat intelligence. Adversarial examples pose risks: "
            "small input perturbations can fool classifiers. LLMs are being used for automated "
            "penetration testing and phishing detection. Privacy-preserving ML techniques like "
            "federated learning reduce data exposure."
        ),
        "url": "internal://docs/cybersecurity_ai",
    },
    {
        "id": "doc_010",
        "title": "Evaluation Metrics for NLP",
        "content": (
            "Common NLP metrics: BLEU and ROUGE for summarisation/translation, F1-score for "
            "classification, BERTScore for semantic similarity, perplexity for language model "
            "quality. Human evaluation remains the gold standard. Benchmarks include MMLU, "
            "HellaSwag, HumanEval, and GLUE/SuperGLUE."
        ),
        "url": "internal://docs/nlp_metrics",
    },
]


_STOPWORDS = {"a", "an", "the", "and", "or", "but", "is", "are", "was", "were", "be", "been", "being", "in", "on", "at", "to", "for", "with", "by", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "from", "up", "down", "out", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "can", "will", "just", "should", "now", "of", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "has", "have", "had", "do", "does", "did", "it", "its", "they", "their", "them", "we", "our", "you", "your", "he", "his", "him", "she", "her", "i", "my", "me", "as"}

def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace, and remove stopwords."""
    tokens = re.findall(r"[a-z]+", text.lower())
    return [t for t in tokens if t not in _STOPWORDS]


def _tf(tokens: list[str]) -> dict[str, float]:
    counts = Counter(tokens)
    total = max(len(tokens), 1)
    return {w: c / total for w, c in counts.items()}


def _idf(word: str, corpus: list[list[str]]) -> float:
    containing = sum(1 for doc in corpus if word in doc)
    return math.log((len(corpus) + 1) / (containing + 1)) + 1


def _score_doc(query_tokens: list[str], doc_tokens: list[str],
               corpus: list[list[str]]) -> float:
    """TF-IDF dot product between query and document."""
    tf = _tf(doc_tokens)
    score = 0.0
    for w in set(query_tokens):
        score += tf.get(w, 0.0) * _idf(w, corpus)
    return score


def run_vector_retrieve(query: str, top_k: int = 3) -> list[dict]:
    """
    Retrieve the top-k documents from the in-memory store using TF-IDF.

    Parameters
    ----------
    query : natural language search string
    top_k : number of documents to return

    Returns
    -------
    list of dicts with keys: query, content, url, title
    Always returns at least an empty list (never raises).
    """
    try:
        query_tokens = _tokenize(query)
        corpus_tokens = [_tokenize(d["content"] + " " + d["title"]) for d in _DOCUMENT_STORE]

        scored = [
            (_score_doc(query_tokens, doc_toks, corpus_tokens), doc)
            for doc_toks, doc in zip(corpus_tokens, _DOCUMENT_STORE)
        ]
        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, doc in scored[:top_k]:
            if score > 0:  # only include docs with non-zero relevance
                results.append({
                    "query":   query,
                    "content": doc["content"],
                    "url":     doc["url"],
                    "title":   doc["title"],
                })
        return results

    except Exception as exc:
        print(f"[VectorStore] Retrieval failed for '{query}': {exc}")
        return []
