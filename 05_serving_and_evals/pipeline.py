"""
Master Pipeline for Production Serving & Evaluation (Project 5).
100% Function-Based: Pure, composable Python functions with zero class boilerplate.

Workflow:
1. Cache Gate: Checks semantic vector cache for previous answers (drops latency from >1s to <5ms).
2. Streaming Generator: Generates token stream or full text response.
3. Automated Evaluation: Evaluates RAG Triad (Faithfulness, Relevance, Precision) on response against context.
4. Telemetry: Emits end-to-end latency, hit rate metrics, and test verdict.
"""

import time
import os
import sys
from typing import Dict, Any, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from .semantic_cache import (
        create_semantic_cache,
        lookup_cache,
        store_in_cache,
        get_cache_metrics,
    )
    from .rag_triad_eval import evaluate_rag_triad
    from .streaming_server import generate_fallback_tokens
except ImportError:
    from semantic_cache import (
        create_semantic_cache,
        lookup_cache,
        store_in_cache,
        get_cache_metrics,
    )
    from rag_triad_eval import evaluate_rag_triad
    from streaming_server import generate_fallback_tokens


GLOBAL_CACHE = create_semantic_cache(similarity_threshold=0.88)


def run_serving_and_eval_pipeline(
    query: str,
    context_chunks: Optional[List[str]] = None,
    ground_truth_keywords: Optional[List[str]] = None,
    custom_cache: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Executes the unified serving & evaluation lifecycle for an enterprise query.
    """
    start_total = time.perf_counter()
    cache = custom_cache if custom_cache is not None else GLOBAL_CACHE
    contexts = context_chunks if context_chunks else [
        "Store return policy allows standard returns within 30 days of purchase with original receipt.",
        "Opened electronics are subject to a 15% restocking fee unless defective.",
    ]

    # Step 1: Semantic Cache lookup
    is_hit, cached_text, sim_score, cache_latency_ms = lookup_cache(cache, query)

    if is_hit and cached_text:
        response_text = cached_text
        cached_flag = True
    else:
        # Cache Miss: Generate response
        context_str = " ".join(contexts)
        tokens = generate_fallback_tokens(query=query, context=context_str)
        response_text = "".join(tokens).strip()
        store_in_cache(cache, query, response_text)
        cached_flag = False

    # Step 2: RAG Triad Evaluation
    eval_result = evaluate_rag_triad(
        query=query,
        answer=response_text,
        retrieved_contexts=contexts,
        ground_truth_keywords=ground_truth_keywords,
    )

    total_latency_ms = (time.perf_counter() - start_total) * 1000

    return {
        "query": query,
        "response": response_text,
        "cached": cached_flag,
        "similarity_score": sim_score,
        "cache_latency_ms": round(cache_latency_ms, 2),
        "total_latency_ms": round(total_latency_ms, 2),
        "evaluation": eval_result,
        "cache_metrics": get_cache_metrics(cache),
    }
