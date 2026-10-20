"""
Semantic Vector Caching Engine for Production AI Serving.
100% Function-Based: Pure, composable Python functions with zero class boilerplate.

Workflow:
1. Embeds incoming queries into normalized dense feature vectors.
2. Performs cosine similarity against previously cached queries.
3. If max similarity >= threshold (default 0.88), returns cached response in <5ms.
4. If cache miss, stores new response and updates metrics (hit rate, latency saved).
"""

import time
import re
from typing import Dict, Any, List, Optional, Tuple
import numpy as np


def tokenize(text: str) -> List[str]:
    """Simple lowercase word tokenizer."""
    return re.findall(r"\b[a-z0-9]+\b", text.lower())


def text_to_vector(text: str, dim: int = 128) -> np.ndarray:
    """Generates a deterministic normalized feature vector for text."""
    tokens = tokenize(text)
    vec = np.zeros(dim, dtype=np.float32)
    if not tokens:
        return vec

    for t in tokens:
        vec[hash(t) % dim] += 1.0
        if len(t) >= 3:
            for j in range(len(t) - 2):
                vec[hash(t[j : j + 3]) % dim] += 0.5

    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def create_semantic_cache(similarity_threshold: float = 0.88, dim: int = 128) -> Dict[str, Any]:
    """
    Initializes a new in-memory semantic cache instance.
    """
    return {
        "similarity_threshold": similarity_threshold,
        "dim": dim,
        "entries": [],  # List of dicts: {"query": str, "vector": np.ndarray, "response": str, "created_at": float}
        "stats": {
            "total_requests": 0,
            "hits": 0,
            "misses": 0,
            "estimated_time_saved_ms": 0.0,
        },
    }


def lookup_cache(
    cache: Dict[str, Any],
    query: str,
    threshold: Optional[float] = None,
) -> Tuple[bool, Optional[str], float, float]:
    """
    Queries semantic cache using cosine similarity.
    Returns: (is_hit, cached_response, similarity_score, latency_ms)
    """
    start_time = time.perf_counter()
    cutoff = threshold if threshold is not None else cache.get("similarity_threshold", 0.88)
    cache["stats"]["total_requests"] += 1

    entries = cache.get("entries", [])
    if not entries:
        cache["stats"]["misses"] += 1
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        return False, None, 0.0, elapsed_ms

    q_vec = text_to_vector(query, dim=cache.get("dim", 128))
    best_score = -1.0
    best_response = None

    for entry in entries:
        score = float(np.dot(entry["vector"], q_vec))
        if score > best_score:
            best_score = score
            best_response = entry["response"]

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    if best_score >= cutoff and best_response is not None:
        cache["stats"]["hits"] += 1
        # Baseline model inference is estimated at ~1200ms
        cache["stats"]["estimated_time_saved_ms"] += max(0.0, 1200.0 - elapsed_ms)
        return True, best_response, round(best_score, 4), elapsed_ms

    cache["stats"]["misses"] += 1
    return False, None, max(0.0, round(best_score, 4)), elapsed_ms


def store_in_cache(
    cache: Dict[str, Any],
    query: str,
    response: str,
) -> Dict[str, Any]:
    """
    Saves a query, its semantic vector, and generated response into cache.
    """
    q_vec = text_to_vector(query, dim=cache.get("dim", 128))
    entry = {
        "query": query,
        "vector": q_vec,
        "response": response,
        "timestamp": time.time(),
    }
    cache["entries"].append(entry)
    return entry


def get_cache_metrics(cache: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns hit rate and latency savings metrics for operational observability.
    """
    stats = cache.get("stats", {})
    total = stats.get("total_requests", 0)
    hits = stats.get("hits", 0)
    hit_rate = round((hits / total) * 100, 2) if total > 0 else 0.0

    return {
        "total_requests": total,
        "hits": hits,
        "misses": stats.get("misses", 0),
        "hit_rate_pct": hit_rate,
        "cached_entries": len(cache.get("entries", [])),
        "estimated_time_saved_ms": round(stats.get("estimated_time_saved_ms", 0.0), 2),
    }
