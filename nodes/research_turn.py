"""Graph stages for query preparation, evidence selection, and response."""

import logging
import re
from time import perf_counter

from state import TurnState
from tools.research_providers import Judgment, QueryPlan, RerankOutputError
from tools.retrieval import SemanticIndex, fuse_ranked, retrieve_lexical, web_candidates


logger = logging.getLogger("uvicorn.error.research")
_MAX_QUERY_LENGTH = 400
_MAX_RETRIES = 1
_CITATION = re.compile(r"\[(\s*\d[\d\s,;–—-]*)\]")


def _log(stage: str, started: float, **metrics: object) -> None:
    logger.info("research_stage=%s elapsed_ms=%.1f %s", stage,
                (perf_counter() - started) * 1000,
                " ".join(f"{key}={value}" for key, value in metrics.items()))


def _queries(values: list[str], searched: list[str], limit: int) -> list[str]:
    seen = {query.casefold() for query in searched}
    queries = []
    for value in values:
        query = " ".join(value.split())
        if not query or len(query) > _MAX_QUERY_LENGTH or query.casefold() in seen:
            continue
        queries.append(query)
        seen.add(query.casefold())
        if len(queries) == limit:
            break
    return queries


def prepare_queries(state: TurnState, prepare) -> TurnState:
    started = perf_counter()
    topic = state["topic"].strip()
    original = state.get("original_queries", {})
    is_retry = bool(original)
    question = state.get("standalone_question", topic)
    searched = state.get("searched_queries", [])
    missing = [original[index] for index in state.get("missing_query_ids", [])]
    retries = state.get("retries", 0) + int(is_retry)
    fallbacks = []
    try:
        plan = QueryPlan.model_validate(prepare(question, state.get("history", [])[-4:], missing))
        queries = _queries(plan.queries, searched, 2 if is_retry else 3)
        if not is_retry:
            question = plan.standalone_question.strip()
            if not question or not queries:
                raise ValueError("No usable search plan")
    except Exception as exc:
        question = state.get("standalone_question", topic)
        queries = [] if is_retry else [topic[:_MAX_QUERY_LENGTH]]
        fallbacks.append("query_rewrite_failed")
        logger.info("research_fallback=query_rewrite_failed reason=%s", type(exc).__name__)

    _log("prepare_queries", started, queries=len(queries), retries=retries)
    return {
        "standalone_question": question,
        "original_queries": original if is_retry else dict(enumerate(queries, 1)),
        "queries": queries,
        "searched_queries": searched + queries,
        "retries": retries,
        "fallbacks": fallbacks,
    }


def after_prepare(state: TurnState) -> list[str] | str:
    return ["retrieve_local", "retrieve_web"] if state["queries"] else "respond"


def retrieve_local(state: TurnState, index: SemanticIndex) -> TurnState:
    started = perf_counter()
    queries = state["queries"]
    lexical = [retrieve_lexical(query) for query in queries]
    try:
        semantic = index.retrieve(queries)
        fallbacks = []
    except Exception as exc:
        semantic = []
        fallbacks = ["semantic_unavailable"]
        logger.info("research_fallback=semantic_unavailable reason=%s", type(exc).__name__)
    _log("retrieve_local", started, lexical=sum(map(len, lexical)),
         semantic=sum(map(len, semantic)))
    return {
        "local_rankings": state.get("local_rankings", []) + lexical + semantic,
        "fallbacks": fallbacks,
    }


def retrieve_web(state: TurnState, web_search) -> TurnState:
    started = perf_counter()
    rankings = []
    failures = 0
    for query in state["queries"]:
        try:
            rankings.append(web_candidates(web_search(query)[:4]))
        except Exception as exc:
            failures += 1
            logger.info("research_fallback=web_search_failed reason=%s", type(exc).__name__)
    _log("retrieve_web", started, results=sum(map(len, rankings)), failures=failures)
    return {
        "web_rankings": state.get("web_rankings", []) + rankings,
        "fallbacks": ["web_search_failed"] if failures else [],
    }


def fuse_candidates(state: TurnState) -> TurnState:
    started = perf_counter()
    rankings = state["local_rankings"] + state["web_rankings"]
    previous_ids = {item["id"] for item in state.get("selected_candidates", [])}
    if previous_ids:
        ranked = fuse_ranked(rankings, limit=None)
        new = [item["id"] for item in ranked if item["id"] not in previous_ids]
        retained = previous_ids | set(new[:12 - len(previous_ids)])
        candidates = [item for item in ranked if item["id"] in retained]
    else:
        candidates = fuse_ranked(rankings)
    _log("fuse_candidates", started, results=len(candidates))
    return {"fused_candidates": candidates}


def rerank_candidates(state: TurnState, rerank) -> TurnState:
    started = perf_counter()
    candidates = state["fused_candidates"]
    queries = state["original_queries"]
    if not candidates:
        _log("rerank_candidates", started, results=0)
        return {"judgments": [], "selected_candidates": [], "rerank_available": True}
    try:
        result = rerank(state["standalone_question"], queries, candidates)
        judgments = [Judgment.model_validate(item) for item in result]
        by_id = {item.candidate_id: item for item in judgments}
        known_ids = {item["id"] for item in candidates}
        if by_id.keys() - known_ids:
            raise RerankOutputError("unknown_candidate_id")
        if len(by_id) != len(judgments):
            raise RerankOutputError("duplicate_candidate_id")
        if known_ids - by_id.keys():
            raise RerankOutputError("missing_candidate_id")

        ignored_coverage = 0
        for index, judgment in enumerate(judgments):
            ids = judgment.covered_query_ids
            covered = list(dict.fromkeys(query_id for query_id in ids if query_id in queries))
            if judgment.relevance < 2:
                covered = []
            if len(covered) != len(ids):
                ignored_coverage += len(ids) - len(covered)
                judgments[index] = judgment.model_copy(update={"covered_query_ids": covered})

        ranked = sorted(candidates, key=lambda item: by_id[item["id"]].relevance, reverse=True)
        selected = [item for item in ranked if by_id[item["id"]].relevance >= 2][:5]
        _log("rerank_candidates", started, results=len(selected), ignored_coverage=ignored_coverage)
        return {"judgments": judgments, "selected_candidates": selected, "rerank_available": True}
    except Exception as exc:
        reason = str(exc) if isinstance(exc, RerankOutputError) else type(exc).__name__
        logger.info("research_fallback=rerank_failed reason=%s", reason)
        prior_judgments = state.get("judgments", [])
        previous_ids = {item["id"] for item in state.get("selected_candidates", [])}
        if previous_ids:
            selected = [item for item in candidates if item["id"] in previous_ids]
        else:
            rejected = {item.candidate_id for item in prior_judgments if item.relevance < 2}
            selected = [item for item in candidates if item["id"] not in rejected][:5]
        _log("rerank_candidates", started, results=len(selected))
        return {
            "judgments": prior_judgments, "selected_candidates": selected,
            "rerank_available": False,
            "fallbacks": ["rerank_failed"],
        }


def check_coverage(state: TurnState) -> TurnState:
    started = perf_counter()
    judgments = state["judgments"]
    selected_ids = {candidate["id"] for candidate in state["selected_candidates"]}
    if not state["rerank_available"]:
        _log("check_coverage", started, coverage="uncertain")
        return {"coverage": "uncertain", "missing_query_ids": state.get("missing_query_ids", [])}

    covered = {index for item in judgments if item.candidate_id in selected_ids
               for index in item.covered_query_ids}
    missing = [index for index in state["original_queries"] if index not in covered]
    if not selected_ids:
        coverage = "insufficient"
    else:
        coverage = "incomplete" if missing else "sufficient"
    _log("check_coverage", started, coverage=coverage, missing=len(missing))
    return {"coverage": coverage, "missing_query_ids": missing}


def after_coverage(state: TurnState) -> str:
    if state["missing_query_ids"] and state["retries"] < _MAX_RETRIES:
        return "prepare_queries"
    return "respond"


def respond(state: TurnState, answer) -> TurnState:
    started = perf_counter()
    snippets = state.get("selected_candidates", [])
    if not snippets:
        _log("respond", started, sources=0)
        return {"final_answer": "I couldn't find enough relevant evidence to answer this question.",
                "sources": []}

    gaps = [state["original_queries"][index] for index in state.get("missing_query_ids", [])]
    uncertain = state["coverage"] == "uncertain"
    answer_gaps = gaps + (["The relevance and coverage of the available evidence are uncertain."]
                          if uncertain else [])
    fallbacks = []
    try:
        final_answer = answer(state["standalone_question"], snippets, answer_gaps).strip()
        valid = {str(index) for index in range(1, len(snippets) + 1)}
        cited = _CITATION.findall(final_answer)
        if not cited or any(number.strip() not in valid for number in cited):
            raise ValueError("Answer did not use valid source citations")
        if re.search(r"\[\s*\d+\s*\][(\[]|(?m:^\s*\[\d+\]:)", final_answer):
            raise ValueError("Answer tried to redefine a numbered citation")
    except Exception as exc:
        logger.info("research_fallback=answer_failed reason=%s", type(exc).__name__)
        final_answer = "I found the sources below, but couldn't produce a cited answer from them."
        fallbacks.append("answer_failed")

    if uncertain:
        final_answer += "\n\nI couldn't verify how fully these sources cover the question."
    if gaps:
        escaped = [re.sub(r"([\\`*_{}\[\]()<>])", r"\\\1", gap) for gap in gaps]
        final_answer += "\n\nEvidence is still missing for these aspects.\n\n"
        final_answer += "\n".join(f"- {gap}" for gap in escaped)
    _log("respond", started, sources=len(snippets))
    return {"final_answer": final_answer, "sources": [item["url"] for item in snippets],
            "fallbacks": fallbacks}
