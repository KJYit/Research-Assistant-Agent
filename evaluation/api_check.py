"""Check deployed API latency with a fresh turn, follow-up, and small concurrent batch."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from math import ceil
import re
from time import perf_counter
from uuid import uuid4

import httpx


_QUESTIONS = (
    "What is retrieval-augmented generation?",
    "How does it reduce hallucinations?",
    "What are common uses for transformer models?",
    "How does reinforcement learning from human feedback work?",
    "What are ROUGE and BERTScore used to evaluate?",
)


def _measure(client: httpx.Client, label: str, topic: str, thread_id: str) -> dict:
    started = perf_counter()
    result = {"case": label, "status": None, "sources": 0, "outcome": "transport_error"}
    try:
        response = client.post("api/research", json={"topic": topic, "thread_id": thread_id})
        result["status"] = response.status_code
        if response.status_code == 200:
            result["outcome"] = "invalid_response"
            try:
                payload = response.json()
            except ValueError:
                payload = None
            if (isinstance(payload, dict) and payload.get("thread_id") == thread_id
                    and isinstance(payload.get("answer"), str) and payload["answer"].strip()
                    and isinstance(payload.get("sources"), list)
                    and all(isinstance(source, str) and source for source in payload["sources"])):
                result["sources"] = len(payload["sources"])
                citations = re.findall(r"\[(\d+)\]", payload["answer"])
                valid_citations = {str(number) for number in range(1, result["sources"] + 1)}
                if not result["sources"]:
                    result["outcome"] = "answer_without_sources"
                elif citations and set(citations) <= valid_citations:
                    result["outcome"] = "cited_answer"
                else:
                    result["outcome"] = "answer_without_valid_citations"
        else:
            result["outcome"] = {429: "rate_limited", 504: "server_timeout"}.get(response.status_code, "http_error")
    except httpx.TimeoutException:
        result["outcome"] = "client_timeout"
    except httpx.RequestError:
        pass

    result["elapsed_seconds"] = round(perf_counter() - started, 2)
    return result


def check_api(base_url: str, concurrency: int = 3) -> dict:
    if concurrency not in (1, 2, 3):
        raise ValueError("Concurrency must be 1, 2, or 3.")
    with httpx.Client(base_url=base_url.rstrip("/") + "/", timeout=95) as client:
        thread_id = uuid4().hex
        results = [_measure(client, "fresh", _QUESTIONS[0], thread_id)]
        if results[0]["outcome"] == "cited_answer":
            results.append(_measure(client, "follow_up", _QUESTIONS[1], thread_id))
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            jobs = [pool.submit(_measure, client, f"concurrent_{index + 1}", question, uuid4().hex)
                    for index, question in enumerate(_QUESTIONS[2:2 + concurrency])]
            results.extend(job.result() for job in jobs)

    durations = sorted(item["elapsed_seconds"] for item in results)
    return {
        "requests": results,
        "expected_requests": concurrency + 2,
        "p95_seconds": durations[ceil(0.95 * len(durations)) - 1],
        "passed": len(results) == concurrency + 2
        and all(item["outcome"] == "cited_answer" for item in results),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_url", help="Base URL of the FastAPI deployment")
    parser.add_argument("--concurrency", type=int, choices=(1, 2, 3), default=3)
    args = parser.parse_args()
    result = check_api(args.base_url, args.concurrency)
    print(json.dumps(result, indent=2))
    return int(not result["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
