"""
Part 5: Production Serving & Evaluation Module.
"""

from .semantic_cache import (
    create_semantic_cache,
    lookup_cache,
    store_in_cache,
    get_cache_metrics,
)
from .rag_triad_eval import (
    evaluate_faithfulness,
    evaluate_answer_relevance,
    evaluate_context_precision,
    evaluate_rag_triad,
    run_eval_suite,
)
from .streaming_server import create_app
from .pipeline import run_serving_and_eval_pipeline

__all__ = [
    "create_semantic_cache",
    "lookup_cache",
    "store_in_cache",
    "get_cache_metrics",
    "evaluate_faithfulness",
    "evaluate_answer_relevance",
    "evaluate_context_precision",
    "evaluate_rag_triad",
    "run_eval_suite",
    "create_app",
    "run_serving_and_eval_pipeline",
]
