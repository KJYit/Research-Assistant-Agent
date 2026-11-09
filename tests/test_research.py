from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from io import StringIO
from threading import Event
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from httpx import ReadTimeout
from langchain_core.messages import AIMessage, HumanMessage

import api
import main
from evaluation.retrieval_benchmark import score
from graph import build_graph, build_turn_graph, invoke_research_turn
from tools.research_providers import Judgment, QueryPlan, ResearchProviders, default_providers
from tools.retrieval import retrieve_lexical


def _prepare(topic, history, missing):
    return QueryPlan(standalone_question=topic, queries=[topic])


def _doc_vectors(texts):
    return [[float(index == item) for index in range(len(texts))] for item in range(len(texts))]


def _query_vectors(texts):
    return [[1.0] + [0.0] * 9 for _ in texts]


def _relevant(question, queries, candidates):
    return [Judgment(candidate_id=item["id"], relevance=3, covered_query_ids=list(queries))
            for item in candidates]


def _answer(question, snippets, gaps):
    return "Supported by the source [1]."


def _providers(**overrides):
    callbacks = dict(prepare=_prepare, embed_documents=_doc_vectors, embed_queries=_query_vectors,
                     web_search=lambda query: [], rerank=_relevant, answer=_answer)
    return ResearchProviders(**(callbacks | overrides))


def _failed(*args):
    raise TimeoutError("unredacted-provider-secret")


class ResearchTurnTests(unittest.TestCase):
    def test_follow_up_is_rewritten_and_deduplicated_before_provider_calls(self):
        calls = []
        history = [HumanMessage(content="What are graph neural networks?"), AIMessage(content="GNNs use edges.")]

        def prepare(topic, previous, missing):
            self.assertEqual(previous, history)
            self.assertEqual(topic, "What are its uses?")
            calls.append("prepare")
            return QueryPlan(standalone_question="What are the uses of graph neural networks?",
                             queries=[" GNN applications ", "gnn applications", "graph edge prediction"])

        def web_search(query):
            calls.append(("web", query))
            return []

        def embed_queries(queries):
            calls.append(("embed", list(queries)))
            return _query_vectors(queries)

        turn = build_turn_graph(_providers(prepare=prepare, web_search=web_search, embed_queries=embed_queries))
        result = turn.invoke({"topic": "What are its uses?", "history": history})
        self.assertEqual(calls[0], "prepare")
        self.assertCountEqual(calls[1:], [
            ("web", "GNN applications"), ("web", "graph edge prediction"),
            ("embed", ["GNN applications", "graph edge prediction"]),
        ])
        self.assertEqual(result["coverage"], "sufficient")

    def test_lexical_exact_match_and_semantic_only_match(self):
        self.assertEqual(retrieve_lexical("BM25 dense passage retrieval")[0]["id"], "doc_002")
        self.assertEqual(retrieve_lexical("qzxqzx"), [])
        indexed = []

        def embed_documents(texts):
            indexed.extend(texts)
            return _doc_vectors(texts)

        def embed_queries(queries):
            doc_index = next(index for index, text in enumerate(indexed) if "oversmoothing" in text.lower())
            return [[float(index == doc_index) for index in range(len(indexed))] for _ in queries]

        turn = build_turn_graph(_providers(embed_documents=embed_documents, embed_queries=embed_queries))
        for _ in range(2):
            result = turn.invoke({"topic": "qzxqzx"})
            self.assertEqual(result["fused_candidates"][0]["id"], "doc_008")
        self.assertEqual(len(indexed), 10, "The second turn should reuse the indexed document vectors.")

    def test_provider_failures_degrade_independently_and_logs_are_redacted(self):
        searched = []

        def web_search(query):
            searched.append(query)
            return _failed()

        with self.assertLogs(level="INFO") as logs:
            turn = build_turn_graph(_providers(prepare=_failed, embed_queries=_failed, web_search=web_search))
            result = turn.invoke({"topic": "  BM25 dense passage retrieval  "})
        self.assertEqual(searched, ["BM25 dense passage retrieval"])
        self.assertEqual(result["sources"][0], "internal://docs/rag_overview")
        joined = "\n".join(logs.output)
        self.assertNotIn("unredacted-provider-secret", joined)
        self.assertNotIn("BM25 dense passage retrieval", joined)

    def test_web_evidence_survives_unavailable_local_providers(self):
        web_result = {"url": "https://example.com/answer", "title": "Answer", "content": "Web evidence"}
        turn = build_turn_graph(_providers(embed_documents=_failed, web_search=lambda query: [web_result]))
        result = turn.invoke({"topic": "qzxqzx"})
        self.assertEqual(result["sources"], [web_result["url"]])

    def test_fusion_deduplicates_normalized_web_urls(self):
        url = "https://example.com/evidence"
        web_results = [
            {"url": "HTTPS://EXAMPLE.COM/evidence#first", "title": "First", "content": "Evidence A"},
            {"url": url, "title": "Second", "content": "Evidence B"},
        ]
        turn = build_turn_graph(_providers(web_search=lambda query: web_results))
        result = turn.invoke({"topic": "qzxqzx"})
        web = [item for item in result["fused_candidates"] if item["url"].lower().startswith("https://")]
        self.assertEqual(len(web), 1)

    def test_unknown_reranker_ids_fall_back_to_fusion(self):
        def malformed(question, queries, candidates):
            return [Judgment(candidate_id="invented", relevance=3, covered_query_ids=[1])]

        turn = build_turn_graph(_providers(rerank=malformed))
        result = turn.invoke({"topic": "BM25 dense passage retrieval"})
        self.assertEqual(result["coverage"], "uncertain")
        self.assertEqual(result["selected_candidates"], result["fused_candidates"][:5])

    def test_invalid_coverage_claims_do_not_discard_valid_relevance_ranking(self):
        def prepare(topic, history, missing):
            return QueryPlan(standalone_question=topic, queries=["BM25 dense passage retrieval", "qzxqzx"])

        def rerank(question, queries, candidates):
            return [Judgment(candidate_id=item["id"], relevance=3 if item["id"] == "doc_002" else 1,
                             covered_query_ids=[1, 99] if item["id"] == "doc_002" else [2])
                    for item in candidates]

        turn = build_turn_graph(_providers(prepare=prepare, rerank=rerank))
        result = turn.invoke({"topic": "How do the two mechanisms compare?"})
        self.assertEqual([item["id"] for item in result["selected_candidates"]], ["doc_002"])
        self.assertEqual(result["coverage"], "incomplete")
        self.assertEqual(result["missing_query_ids"], [2])
        self.assertNotIn("rerank_failed", result["fallbacks"])

    def test_zero_relevance_is_insufficient_and_skips_answer_generation(self):
        answer_calls = []

        def irrelevant(question, queries, candidates):
            return [Judgment(candidate_id=item["id"], relevance=0, covered_query_ids=[]) for item in candidates]

        def unexpected_answer(*args):
            answer_calls.append(args)
            return "Unexpected response"

        turn = build_turn_graph(_providers(rerank=irrelevant, answer=unexpected_answer))
        result = turn.invoke({"topic": "What is the elevation of Mount Kinabalu?"})
        self.assertEqual(result["coverage"], "insufficient")
        self.assertEqual(result["sources"], [])
        self.assertEqual(answer_calls, [])
        self.assertIn("evidence", result["final_answer"].lower())

    def test_retry_is_bounded_preserves_evidence_and_keeps_initial_query_ids(self):
        attempts, searched, assessed = [], [], []

        def prepare(topic, history, missing):
            attempts.append(list(missing))
            queries = ["first query", "second query"] if not missing else ["First query", "another query", "new query"]
            return QueryPlan(standalone_question=topic, queries=queries)

        def web_search(query):
            searched.append(query)
            return [{"url": "https://example.com/first", "title": "First", "content": "Earlier evidence"}] if query == "first query" else []

        def rerank(question, queries, candidates):
            assessed.append(queries.copy())
            return [Judgment(candidate_id=item["id"], relevance=3, covered_query_ids=[1]) for item in candidates]

        turn = build_turn_graph(_providers(prepare=prepare, web_search=web_search, rerank=rerank))
        result = turn.invoke({"topic": "Compare two approaches"})
        self.assertEqual(len(attempts), 2)
        self.assertEqual(len(assessed), 2)
        self.assertEqual(assessed[0], {1: "first query", 2: "second query"})
        self.assertEqual(assessed[1], assessed[0])
        self.assertCountEqual(searched, ["first query", "second query", "another query", "new query"])
        self.assertIn("https://example.com/first", result["sources"])
        self.assertEqual(result["missing_query_ids"], [2])

    def test_retry_reranker_failure_keeps_previous_validated_evidence(self):
        retained = "https://example.com/qzxfirst/3"
        rerank_calls = []

        def prepare(topic, history, missing):
            queries = ["neural network models", "artificial intelligence"] if missing else ["qzxfirst", "qzxsecond"]
            return QueryPlan(standalone_question=topic, queries=queries)

        def web_search(query):
            slug = query.replace(" ", "-")
            return [{"url": f"https://example.com/{slug}/{index}", "title": "Evidence", "content": "Text"}
                    for index in range(4)]

        def rerank(question, queries, candidates):
            rerank_calls.append([item["url"] for item in candidates])
            if len(rerank_calls) == 2:
                return _failed()
            return [Judgment(candidate_id=item["id"], relevance=3 if item["url"] == retained else 0,
                             covered_query_ids=[1] if item["url"] == retained else [])
                    for item in candidates]

        turn = build_turn_graph(_providers(prepare=prepare, rerank=rerank, web_search=web_search))
        result = turn.invoke({"topic": "Compare the two subjects"})
        self.assertEqual(len(rerank_calls), 2)
        self.assertLessEqual(len(rerank_calls[1]), 12)
        self.assertIn(retained, rerank_calls[1])
        self.assertEqual(result["sources"], [retained])
        self.assertEqual(result["coverage"], "uncertain")
        self.assertEqual(result["missing_query_ids"], [2])

    def test_retry_reranker_failure_does_not_resurrect_rejected_evidence(self):
        rerank_calls = []

        def prepare(topic, history, missing):
            query = "qzxsecond" if missing else "qzxfirst"
            return QueryPlan(standalone_question=topic, queries=[query])

        def rerank(question, queries, candidates):
            rerank_calls.append({item["id"] for item in candidates})
            if len(rerank_calls) == 2:
                return _failed()
            return [Judgment(candidate_id=item["id"], relevance=0, covered_query_ids=[]) for item in candidates]

        web_results = [{"url": "https://example.com/tangential", "title": "Tangential", "content": "Off-topic"}]
        turn = build_turn_graph(_providers(prepare=prepare, rerank=rerank, web_search=lambda query: web_results))
        result = turn.invoke({"topic": "What is the elevation of Mount Kinabalu?"})
        self.assertEqual(len(rerank_calls), 2)
        self.assertTrue(rerank_calls[0])
        self.assertEqual(rerank_calls[0], rerank_calls[1])
        self.assertEqual(result["selected_candidates"], [])
        self.assertEqual(result["sources"], [])
        self.assertIn("evidence", result["final_answer"].lower())

    def test_answer_citations_cannot_reference_unselected_evidence(self):
        turn = build_turn_graph(_providers(answer=lambda *args: "Valid [1] and invalid [99]."))
        result = turn.invoke({"topic": "BM25 dense passage retrieval"})
        self.assertNotIn("[99]", result["final_answer"])
        self.assertEqual(result["sources"], [candidate["url"] for candidate in result["selected_candidates"]])

    def test_coverage_uses_only_snippets_selected_for_the_answer(self):
        def prepare(topic, history, missing):
            return QueryPlan(standalone_question=topic, queries=["neural network models", "artificial intelligence"])

        def rerank(question, queries, candidates):
            return [Judgment(candidate_id=item["id"], relevance=3 if index < 5 else 2,
                             covered_query_ids=[1] if index < 5 else [2])
                    for index, item in enumerate(candidates)]

        web_results = [{"url": f"https://example.com/evidence/{index}", "title": "Evidence", "content": "Text"}
                       for index in range(4)]
        turn = build_turn_graph(_providers(prepare=prepare, rerank=rerank, web_search=lambda query: web_results))
        result = turn.invoke({"topic": "Compare the two subjects"})
        self.assertGreater(len(result["fused_candidates"]), 5)
        self.assertEqual(len(result["selected_candidates"]), 5)
        self.assertEqual(result["missing_query_ids"], [2])
        self.assertEqual(result["coverage"], "incomplete")


class ResearchSessionTests(unittest.TestCase):
    def test_threads_keep_own_history_and_no_private_checkpoint_channels(self):
        observed = {}

        def prepare(topic, history, missing):
            observed[topic] = [message.content for message in history]
            return _prepare(topic, history, missing)

        graph = build_graph(_providers(prepare=prepare))
        invoke_research_turn("first", "one", graph=graph)
        invoke_research_turn("independent", "two", graph=graph)
        invoke_research_turn("follow up", "one", graph=graph)
        self.assertEqual(observed["first"], [])
        self.assertEqual(observed["independent"], [])
        self.assertEqual(observed["follow up"], ["first", "Supported by the source [1]."])
        for thread_id in ("one", "two"):
            config = {"configurable": {"thread_id": thread_id}}
            state = graph.get_state(config).values
            self.assertLessEqual(set(state), {"topic", "messages", "final_answer", "sources"})

    def test_same_thread_turns_are_serialized(self):
        first_entered, second_started, second_entered, release = Event(), Event(), Event(), Event()

        def prepare(topic, history, missing):
            if topic == "first":
                first_entered.set()
                self.assertTrue(release.wait(5))
            else:
                second_entered.set()
            return _prepare(topic, history, missing)

        graph = build_graph(_providers(prepare=prepare))

        def second_turn():
            second_started.set()
            invoke_research_turn("second", "thread", graph=graph)

        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(invoke_research_turn, "first", "thread", graph=graph)
            try:
                self.assertTrue(first_entered.wait(5))
                second = executor.submit(second_turn)
                self.assertTrue(second_started.wait(5))
                self.assertFalse(second_entered.wait(0.1))
            finally:
                release.set()
            first.result(timeout=5)
            second.result(timeout=5)
        self.assertTrue(second_entered.is_set())

    def test_api_contract_and_empty_topic(self):
        with patch("graph.research_graph", build_graph(_providers())), TestClient(api.app) as client:
            response = client.post("/api/research", json={"topic": "BM25 retrieval", "thread_id": "api-test"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(set(response.json()), {"answer", "sources", "thread_id"})
            self.assertEqual(response.json()["thread_id"], "api-test")
            self.assertEqual(response.json()["answer"], "Supported by the source [1].")
            empty = client.post("/api/research", json={"topic": "  ", "thread_id": "api-test"})
            self.assertEqual(empty.status_code, 400)

    def test_cli_entry_point(self):
        with (patch("graph.research_graph", build_graph(_providers())),
              patch.dict("os.environ", {"GOOGLE_API_KEY": "offline", "TAVILY_API_KEY": "offline"}),
              patch("builtins.input", side_effect=["BM25 retrieval", "quit"]),
              redirect_stdout(StringIO()) as output):
            main.main()
        self.assertIn("Supported by the source [1].", output.getvalue())


class RetrievalMetricTests(unittest.TestCase):
    def test_recall_cutoff_and_graded_ordering(self):
        grades = {"high": 3, "low": 1}
        self.assertEqual(score(["high", "low"], grades), (1.0, 1.0))
        recall, ndcg = score(["low", "x", "y", "z", "w", "high"], grades)
        self.assertEqual(recall, 0.5)
        self.assertLess(ndcg, score(["high"], grades)[1])
        self.assertEqual(score(["high", "high"], grades), score(["high"], grades))


class ResearchProviderTests(unittest.TestCase):
    def test_default_reranker_retries_transient_timeout_and_maps_ordinals(self):
        invoke = Mock()
        timeout = ReadTimeout("transient")
        timeout.__cause__ = OSError("lower-level transport error")

        def structured_output(schema):
            if invoke.side_effect is None:
                invoke.side_effect = [timeout, schema.model_validate({
                    "judgments": [
                        {"candidate_id": 2, "relevance": 3, "covered_query_ids": [1]},
                        {"candidate_id": 1, "relevance": 0, "covered_query_ids": []},
                    ],
                })]
            return Mock(invoke=invoke)

        candidates = [
            {"id": "web:https://example.com/test", "url": "https://example.com/test", "title": "Web", "content": "Text"},
            {"id": "doc_002", "url": "internal://docs/rag_overview", "title": "RAG", "content": "Evidence"},
        ]
        with patch("tools.research_providers.ChatGoogleGenerativeAI") as chat:
            chat.return_value.with_structured_output.side_effect = structured_output
            judgments = default_providers().rerank("What is RAG?", {1: "RAG overview"}, candidates)

        self.assertEqual(invoke.call_count, 2)
        self.assertEqual(invoke.call_args_list[0], invoke.call_args_list[1])
        self.assertEqual([item.candidate_id for item in judgments], ["doc_002", "web:https://example.com/test"])


if __name__ == "__main__":
    unittest.main()
