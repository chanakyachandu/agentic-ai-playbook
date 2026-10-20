"""
Test Suite for Part 5: Production Serving, Streaming, Caching & Evaluation.
Covers:
1. Semantic Vector Cache (hit, miss, threshold, operational metrics).
2. RAG Triad Evaluation (faithfulness, answer relevance, context precision, batch suite).
3. FastAPI Microservice & SSE Token Streaming (health, chat, stream endpoints).
4. Master Serving & Evaluation Pipeline.
"""

import json
import os
import sys
import importlib
import pytest
from starlette.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

cache_mod = importlib.import_module("05_serving_and_evals.semantic_cache")
eval_mod = importlib.import_module("05_serving_and_evals.rag_triad_eval")
server_mod = importlib.import_module("05_serving_and_evals.streaming_server")
pipeline_mod = importlib.import_module("05_serving_and_evals.pipeline")

create_semantic_cache = cache_mod.create_semantic_cache
lookup_cache = cache_mod.lookup_cache
store_in_cache = cache_mod.store_in_cache
get_cache_metrics = cache_mod.get_cache_metrics

evaluate_faithfulness = eval_mod.evaluate_faithfulness
evaluate_answer_relevance = eval_mod.evaluate_answer_relevance
evaluate_context_precision = eval_mod.evaluate_context_precision
evaluate_rag_triad = eval_mod.evaluate_rag_triad
run_eval_suite = eval_mod.run_eval_suite

create_app = server_mod.create_app
run_serving_and_eval_pipeline = pipeline_mod.run_serving_and_eval_pipeline


# ==============================================================================
# 1. Semantic Cache Tests
# ==============================================================================
def test_semantic_cache_hit_and_miss():
    cache = create_semantic_cache(similarity_threshold=0.85)

    # 1. Empty cache miss
    hit, resp, score, lat = lookup_cache(cache, "What is the return policy for laptops?")
    assert not hit
    assert resp is None
    assert score == 0.0
    assert lat >= 0.0

    # 2. Store response
    store_in_cache(
        cache,
        query="What is the return policy for laptops?",
        response="Laptops can be returned within 30 days with a 15% restocking fee.",
    )

    # 3. Exact match hit
    hit, resp, score, _ = lookup_cache(cache, "What is the return policy for laptops?")
    assert hit
    assert resp == "Laptops can be returned within 30 days with a 15% restocking fee."
    assert score >= 0.99

    # 4. Semantically close query hit
    hit_close, resp_close, score_close, _ = lookup_cache(
        cache, "What is the return policy on laptops?"
    )
    assert hit_close
    assert resp_close is not None
    assert score_close >= 0.85

    # 5. Completely different query miss
    hit_diff, resp_diff, score_diff, _ = lookup_cache(cache, "What is the weather in Tokyo?")
    assert not hit_diff
    assert resp_diff is None
    assert score_diff < 0.85


def test_semantic_cache_metrics():
    cache = create_semantic_cache(similarity_threshold=0.88)
    store_in_cache(cache, "return window", "30 days")

    lookup_cache(cache, "return window")        # Hit
    lookup_cache(cache, "random question 123")  # Miss

    metrics = get_cache_metrics(cache)
    assert metrics["total_requests"] == 2
    assert metrics["hits"] == 1
    assert metrics["misses"] == 1
    assert metrics["hit_rate_pct"] == 50.0
    assert metrics["cached_entries"] == 1
    assert metrics["estimated_time_saved_ms"] > 0.0


# ==============================================================================
# 2. RAG Triad Evaluation Tests
# ==============================================================================
def test_faithfulness_metric():
    context = ["Laptops must be returned within 30 days in original packaging."]
    grounded_answer = "You can return laptops within 30 days in original packaging."
    hallucinated_answer = "Laptops have lifetime warranties with free international flights."

    score_grounded = evaluate_faithfulness(grounded_answer, context)
    score_hallucinated = evaluate_faithfulness(hallucinated_answer, context)

    assert score_grounded >= 0.70
    assert score_hallucinated < 0.35


def test_answer_relevance_metric():
    query = "What is the return window for clothing?"
    relevant_answer = "The return window for clothing items is 30 days from purchase date."
    irrelevant_answer = "Quantum entanglement describes particles interacting across vast distances."

    rel_score = evaluate_answer_relevance(query, relevant_answer)
    irrel_score = evaluate_answer_relevance(query, irrelevant_answer)

    assert rel_score >= 0.60
    assert irrel_score < 0.35


def test_context_precision_metric():
    keywords = ["laptop", "warranty", "30 days"]
    perfect_contexts = [
        "Laptop warranty covers repairs for 30 days after delivery.",
        "General store clothing policy.",
        "Kitchen utensils disclaimer.",
    ]
    poor_contexts = [
        "General store clothing policy.",
        "Kitchen utensils disclaimer.",
        "Laptop warranty covers repairs for 30 days after delivery.",
    ]

    p_high = evaluate_context_precision(keywords, perfect_contexts)
    p_low = evaluate_context_precision(keywords, poor_contexts)

    assert p_high > p_low
    assert p_high >= 0.80


def test_rag_triad_composite_and_suite():
    res = evaluate_rag_triad(
        query="How long do I have to return an opened drone?",
        answer="You have 30 days to return an opened drone subject to restocking fees.",
        retrieved_contexts=[
            "Drone return policy: Drones must be returned within 30 days with original packaging."
        ],
        ground_truth_keywords=["drone", "30 days", "return"],
    )
    assert "faithfulness" in res
    assert "answer_relevance" in res
    assert "context_precision" in res
    assert res["composite_score"] > 0.50
    assert res["passed"] is True

    # Test batch runner
    suite_res = run_eval_suite([
        {
            "query": "How long do I have to return an opened drone?",
            "answer": "You have 30 days to return an opened drone.",
            "contexts": ["Drones must be returned within 30 days."],
            "ground_truth_keywords": ["drone", "30 days"],
        }
    ])
    assert suite_res["total_evaluated"] == 1
    assert suite_res["passed_cases"] == 1
    assert suite_res["suite_passed"] is True


# ==============================================================================
# 3. FastAPI & SSE Streaming Tests
# ==============================================================================
def test_fastapi_health_endpoint():
    app = create_app()
    client = TestClient(app)

    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "agentic-ai-serving"


def test_fastapi_chat_and_cache_caching():
    app = create_app(cache_threshold=0.85)
    client = TestClient(app)

    # 1. First call -> Cache Miss
    payload = {"query": "Explain store refund duration"}
    r1 = client.post("/chat", json=payload)
    assert r1.status_code == 200
    d1 = r1.json()
    assert d1["cached"] is False
    assert len(d1["response"]) > 0

    # 2. Second identical call -> Cache Hit
    r2 = client.post("/chat", json=payload)
    assert r2.status_code == 200
    d2 = r2.json()
    assert d2["cached"] is True
    assert d2["response"] == d1["response"]


def test_fastapi_stream_sse_endpoint():
    app = create_app()
    client = TestClient(app)

    payload = {"query": "Explain restocking fee rules"}
    with client.stream("POST", "/chat/stream", json=payload) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        events = list(response.iter_lines())
        data_lines = [e for e in events if e.startswith("data: ")]
        assert len(data_lines) > 0

        # Verify parsed payloads contain streaming tokens
        first_payload = json.loads(data_lines[0].replace("data: ", ""))
        assert "token" in first_payload
        assert first_payload["status"] == "streaming"

        # Verify last payload has completion status
        last_payload = json.loads(data_lines[-1].replace("data: ", ""))
        assert last_payload["status"] == "done"
        assert "latency_ms" in last_payload


# ==============================================================================
# 4. Master Pipeline Test
# ==============================================================================
def test_serving_eval_pipeline():
    test_cache = create_semantic_cache()
    out = run_serving_and_eval_pipeline(
        query="What is the policy on 30 day returns?",
        custom_cache=test_cache,
    )
    assert out["cached"] is False
    assert out["total_latency_ms"] >= 0.0
    assert "evaluation" in out
    assert out["evaluation"]["faithfulness"] >= 0.50

    # Repeat query -> should be cached
    out_cached = run_serving_and_eval_pipeline(
        query="What is the policy on 30 day returns?",
        custom_cache=test_cache,
    )
    assert out_cached["cached"] is True
