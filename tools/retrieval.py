"""Bundled research corpus and ranking functions."""

import math
import re
from collections import Counter, defaultdict
from collections.abc import Callable
from threading import Lock
from urllib.parse import urlsplit, urlunsplit

from .research_budget import within_budget
from .research_providers import Candidate


DOCUMENTS: list[Candidate] = [
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
    for w in sorted(set(query_tokens)):
        score += tf.get(w, 0.0) * _idf(w, corpus)
    return score


_CORPUS_TOKENS = [_tokenize(doc["content"] + " " + doc["title"]) for doc in DOCUMENTS]


def retrieve_lexical(query: str, top_k: int = 5) -> list[Candidate]:
    """Rank the bundled corpus with the original TF-IDF scorer."""
    tokens = _tokenize(query)
    scored = [
        (_score_doc(tokens, document_tokens, _CORPUS_TOKENS), doc)
        for doc, document_tokens in zip(DOCUMENTS, _CORPUS_TOKENS)
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    return [doc.copy() for score, doc in scored[:top_k] if score > 0]


def _unit_vectors(vectors: list[list[float]], expected: int) -> list[list[float]]:
    if len(vectors) != expected:
        raise ValueError("Unexpected embedding count")
    width = len(vectors[0])
    normalized = []
    for vector in vectors:
        norm = math.hypot(*vector)
        if len(vector) != width or not math.isfinite(norm) or norm == 0:
            raise ValueError("Invalid embedding vector")
        normalized.append([value / norm for value in vector])
    return normalized


class SemanticIndex:
    """Cache immutable corpus vectors within a graph instance."""

    def __init__(self, embed_documents: Callable, embed_queries: Callable) -> None:
        self._embed_documents = embed_documents
        self._embed_queries = embed_queries
        self._document_vectors: list[list[float]] | None = None
        self._lock = Lock()

    def retrieve(self, queries: list[str], top_k: int = 5) -> list[list[Candidate]]:
        with within_budget(self._lock):
            if self._document_vectors is None:
                texts = [f"{doc['title']}\n{doc['content']}" for doc in DOCUMENTS]
                self._document_vectors = _unit_vectors(self._embed_documents(texts), len(texts))
            documents = self._document_vectors

        vectors = _unit_vectors(self._embed_queries(queries), len(queries))
        if len(vectors[0]) != len(documents[0]):
            raise ValueError("Query and document embedding dimensions differ")
        results = []
        for query in vectors:
            scored = [
                (sum(a * b for a, b in zip(query, document)), doc)
                for document, doc in zip(documents, DOCUMENTS)
            ]
            scored.sort(key=lambda item: item[0], reverse=True)
            results.append([doc.copy() for score, doc in scored[:top_k] if score > 0])
        return results


def _normalize_url(url: str) -> str:
    parsed = urlsplit(url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username is not None:
        raise ValueError("Invalid web URL")
    port = parsed.port
    host = parsed.hostname.lower()
    if ":" in host:
        host = f"[{host}]"
    if port is not None and (parsed.scheme, port) not in {("http", 80), ("https", 443)}:
        host = f"{host}:{port}"
    return urlunsplit((parsed.scheme, host, parsed.path or "/", parsed.query, ""))


def web_candidates(items: list[dict]) -> list[Candidate]:
    candidates: list[Candidate] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        url, title, content = item.get("url"), item.get("title"), item.get("content")
        if not isinstance(url, str) or not isinstance(content, str) or not content.strip():
            continue
        try:
            normalized = _normalize_url(url)
        except ValueError:
            continue
        candidates.append({
            "id": f"web:{normalized}", "url": normalized,
            "title": title[:200] if isinstance(title, str) else "",
            "content": content.strip()[:1000],
        })
    return candidates


def fuse_ranked(ranked_lists: list[list[Candidate]], limit: int | None = 12) -> list[Candidate]:
    """Merge rankings with RRF, counting a candidate at most once per list."""
    candidates: dict[str, Candidate] = {}
    scores: dict[str, float] = defaultdict(float)
    for ranking in ranked_lists:
        seen: set[str] = set()
        for candidate in ranking:
            key = candidate["url"] if candidate["id"].startswith("web:") else candidate["id"]
            if key in seen:
                continue
            seen.add(key)
            candidates.setdefault(key, candidate)
            scores[key] += 1 / (60 + len(seen))
    keys = sorted(scores, key=scores.__getitem__, reverse=True)
    return [candidates[key] for key in keys[:limit]]
