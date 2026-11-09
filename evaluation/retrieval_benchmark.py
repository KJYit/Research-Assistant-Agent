"""Compare local retrieval stages, or explicitly smoke-test the live providers."""

import argparse
from collections import Counter
from dataclasses import replace
from functools import wraps
import hashlib
import json
import math
import os
from pathlib import Path
import re
from statistics import mean
from time import perf_counter

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

from graph import build_turn_graph
from tools.research_providers import Judgment, QueryPlan, ResearchProviders, default_providers
from tools.retrieval import retrieve_lexical


CASES = Path(__file__).with_name("cases.json")
TOP_K = 5
MODES = ("lexical", "hybrid_before_rerank", "full_retrieval")


def score(ranking, grades):
    hits = dict.fromkeys(ranking[:TOP_K])
    recall = sum(doc_id in grades for doc_id in hits) / len(grades)

    def dcg(gains):
        return sum((2 ** grade - 1) / math.log2(rank + 2) for rank, grade in enumerate(gains))

    seen = set()
    gains = []
    for doc_id in ranking[:TOP_K]:
        gains.append(grades.get(doc_id, 0) if doc_id not in seen else 0)
        seen.add(doc_id)
    return recall, dcg(gains) / dcg(sorted(grades.values(), reverse=True)[:TOP_K])


def _fake_vectors(texts):
    vectors = []
    for text in texts:
        vector = [0.0] * 128
        for word in re.findall(r"[a-z0-9]{3,}", text.lower()):
            for offset in range(len(word) - 2):
                bucket = hashlib.blake2s(word[offset:offset + 3].encode(), digest_size=1).digest()[0]
                vector[bucket % len(vector)] += 1
        vectors.append(vector)
    return vectors


def _fake_prepare(topic, history, missing):
    context = next((str(message.content) for message in reversed(history) if isinstance(message, HumanMessage)), "")
    standalone = f"{context} {topic}".strip()
    return QueryPlan(standalone_question=standalone, queries=[standalone])


def _fake_rerank(question, queries, candidates):
    question_words = set(re.findall(r"[a-z0-9]{5,}", question.lower()))
    judgments = []
    for candidate in candidates:
        doc_words = set(re.findall(r"[a-z0-9]{5,}", f"{candidate['title']} {candidate['content']}".lower()))
        overlap = len(doc_words & question_words)
        judgments.append(Judgment(
            candidate_id=candidate["id"], relevance=min(overlap, 3),
            covered_query_ids=list(queries) if overlap >= 2 else [],
        ))
    return judgments


def _fake_answer(question, snippets, gaps):
    return "Evaluation stub [1]"


def _offline_providers():
    return ResearchProviders(
        prepare=_fake_prepare, embed_documents=_fake_vectors, embed_queries=_fake_vectors,
        web_search=lambda query: [], rerank=_fake_rerank, answer=_fake_answer,
    )


def _watch_providers(providers, names):
    successes, failures = Counter(), Counter()

    def observe(name, callback):
        @wraps(callback)
        def call(*args, **kwargs):
            try:
                result = callback(*args, **kwargs)
            except Exception:
                failures[name] += 1
                raise
            if result:
                successes[name] += 1
            return result
        return call

    watched = {name: observe(name, getattr(providers, name)) for name in names}
    return replace(providers, **watched), successes, failures


def _ids(candidates):
    return [candidate["id"] for candidate in candidates[:TOP_K]]


def _measure(turn, case):
    start = perf_counter()
    lexical = _ids(retrieve_lexical(case["topic"], top_k=TOP_K))
    lexical_ms = (perf_counter() - start) * 1000
    history = [
        (HumanMessage if index % 2 == 0 else AIMessage)(content=text)
        for index, text in enumerate(case.get("history", []))
    ]
    state = {}
    fallbacks = []
    first_fused = None
    start = perf_counter()
    for event in turn.stream({"topic": case["topic"], "history": history}, stream_mode="updates"):
        for update in event.values():
            if not update:
                continue
            state.update(update)
            fallbacks.extend(update.get("fallbacks", []))
            if first_fused is None and "fused_candidates" in update:
                first_fused = (_ids(update["fused_candidates"]), (perf_counter() - start) * 1000)
    full_ms = (perf_counter() - start) * 1000
    if first_fused is None:
        raise RuntimeError("The turn graph did not report a fusion stage.")
    rankings = {
        "lexical": (lexical, lexical_ms),
        "hybrid_before_rerank": first_fused,
        "full_retrieval": (_ids(state["selected_candidates"]), full_ms),
    }
    return rankings, fallbacks


def _summarize(results, mode):
    latencies = sorted(measurement[mode][1] for _, measurement in results)
    scores = [score(measurement[mode][0], case["relevance"]) for case, measurement in results if case["relevance"]]
    no_answer = [measurement[mode][0] for case, measurement in results if not case["relevance"]]
    return {
        "recall_at_5": mean(recall for recall, _ in scores),
        "ndcg_at_5": mean(ndcg for _, ndcg in scores),
        "mean_ms": round(mean(latencies), 2),
        "p95_ms": round(latencies[math.ceil(0.95 * len(latencies)) - 1], 2),
        "out_of_corpus_with_results": sum(bool(ranking) for ranking in no_answer),
        "out_of_corpus_total": len(no_answer),
    }


def _regressed_cases(results):
    regressions = []
    for case, measurement in results:
        if not case["relevance"]:
            continue
        grades = case["relevance"]
        baseline_recall, baseline_ndcg = score(measurement["lexical"][0], grades)
        full_recall, full_ndcg = score(measurement["full_retrieval"][0], grades)
        recall_delta, ndcg_delta = full_recall - baseline_recall, full_ndcg - baseline_ndcg
        if recall_delta < 0 or ndcg_delta < 0:
            regressions.append({
                "id": case["id"],
                "recall_at_5_delta": round(recall_delta, 4),
                "ndcg_at_5_delta": round(ndcg_delta, 4),
            })
    return regressions


def _benchmark(live):
    names = ("prepare", "embed_documents", "embed_queries", "rerank")
    if live:
        providers = replace(default_providers(), web_search=lambda query: [], answer=_fake_answer)
        providers, successes, failures = _watch_providers(providers, names)
    else:
        providers = _offline_providers()
    turn = build_turn_graph(providers)
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    results = []
    fallbacks = Counter()
    for case in cases:
        measurements, case_fallbacks = _measure(turn, case)
        results.append((case, measurements))
        fallbacks.update(case_fallbacks)
    metrics = {mode: _summarize(results, mode) for mode in MODES}
    matched_baseline = all(metrics["full_retrieval"][name] >= metrics["lexical"][name]
                           for name in ("recall_at_5", "ndcg_at_5"))
    providers_ok = not live or (not failures and not fallbacks and all(successes[name] for name in names))
    report = {
        "mode": "live_local_corpus" if live else "offline_simulation",
        "cases": len(cases),
        "scored_cases": sum(bool(case["relevance"]) for case in cases),
        "groups": dict(Counter(case["group"] for case in cases)),
        "comparison": {
            "lexical": "Raw-topic TF-IDF over the bundled corpus",
            "hybrid_before_rerank": "Rewritten queries through first-round rank fusion",
            "full_retrieval": "Final evidence after reranking and any coverage retry, with stubbed response generation",
        },
        "metrics": metrics,
        "passes_quality_gate": (providers_ok and matched_baseline) if live else None,
    }
    if live:
        report["provider_failures"] = dict(failures)
        report["graph_fallbacks"] = dict(fallbacks)
        report["providers_with_successful_results"] = sorted(successes)
        report["quality_gate_status"] = "provider_error" if not providers_ok else (
            "passed" if matched_baseline else "ranking_regression"
        )
        if providers_ok and not matched_baseline:
            report["regressed_cases"] = _regressed_cases(results)
    print(json.dumps(report, indent=2))
    if not providers_ok:
        return 2
    return int(live and not matched_baseline)


def _smoke():
    names = ("prepare", "embed_documents", "embed_queries", "web_search", "rerank", "answer")
    providers, successes, failures = _watch_providers(default_providers(), names)
    result = build_turn_graph(providers).invoke({"topic": "How does retrieval-augmented generation use a knowledge base?"})
    passed = (not failures and not result.get("fallbacks") and all(successes[name] for name in names)
              and bool(result["sources"]) and result["coverage"] != "uncertain")
    print(json.dumps({
        "passed": passed,
        "providers_with_successful_results": sorted(successes),
        "provider_failures": dict(failures),
        "graph_fallbacks": result.get("fallbacks", []),
        "source_count": len(result["sources"]),
    }, indent=2))
    return int(not passed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--live", action="store_true", help="Use Gemini on the fixed local corpus.")
    mode.add_argument("--smoke", action="store_true", help="Run one real turn with Gemini and Tavily.")
    parser.add_argument("--env-file", type=Path, help="Load this dotenv file in live modes.")
    args = parser.parse_args()
    if args.live or args.smoke:
        load_dotenv(args.env_file if args.env_file else CASES.parents[1] / ".env")
        missing = []
        if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
            missing.append("GOOGLE_API_KEY or GEMINI_API_KEY")
        if args.smoke and not os.getenv("TAVILY_API_KEY"):
            missing.append("TAVILY_API_KEY")
        if missing:
            parser.error(f"Missing required environment variables: {', '.join(missing)}")
    return _smoke() if args.smoke else _benchmark(args.live)


if __name__ == "__main__":
    raise SystemExit(main())
