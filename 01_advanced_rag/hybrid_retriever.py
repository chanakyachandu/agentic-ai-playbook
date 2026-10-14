"""
Hybrid Retrieval Engine for Enterprise RAG.
100% Function-Based: No classes, pure readable Python functions.
Combines BM25 (Lexical) and Dense Vectors with Reciprocal Rank Fusion (RRF).
"""

import math
import re
from typing import List, Dict, Any
import numpy as np


def tokenize(text: str) -> List[str]:
    """Simple lowercase word tokenizer."""
    return re.findall(r"\b[a-z0-9]+\b", text.lower())


# ==============================================================================
# 1. BM25 Lexical Keyword Search
# ==============================================================================
def bm25_search(
    query: str, chunks: List[Dict[str, Any]], k1: float = 1.5, b: float = 0.75
) -> List[Dict[str, Any]]:
    """
    Computes Okapi BM25 ranking for a query across a list of text chunks.
    Returns chunks ranked by BM25 score.
    """
    corpus_size = len(chunks)
    if corpus_size == 0:
        return []

    doc_tokens_list = [tokenize(c["text"]) for c in chunks]
    doc_lens = [len(tokens) for tokens in doc_tokens_list]
    avg_doc_len = sum(doc_lens) / max(corpus_size, 1)

    # Document frequencies
    doc_freqs: Dict[str, int] = {}
    for tokens in doc_tokens_list:
        for t in set(tokens):
            doc_freqs[t] = doc_freqs.get(t, 0) + 1

    # IDF scores
    query_tokens = tokenize(query)
    scores = [0.0] * corpus_size

    for term in query_tokens:
        if term not in doc_freqs:
            continue
        df = doc_freqs[term]
        idf = math.log(1.0 + (corpus_size - df + 0.5) / (df + 0.5))

        for i, tokens in enumerate(doc_tokens_list):
            tf = tokens.count(term)
            if tf == 0:
                continue
            doc_len = doc_lens[i]
            numerator = tf * (k1 + 1.0)
            denominator = tf + k1 * (1.0 - b + b * (doc_len / avg_doc_len))
            scores[i] += idf * (numerator / denominator)

    # Attach scores and sort
    results = []
    for i, c in enumerate(chunks):
        item = dict(c)
        item["bm25_score"] = round(scores[i], 4)
        results.append(item)

    results.sort(key=lambda x: x["bm25_score"], reverse=True)
    return results


# ==============================================================================
# 2. Dense Semantic Search (Zero-Dependency Vector Embeddings)
# ==============================================================================
def text_to_vector(text: str, dim: int = 128) -> np.ndarray:
    """Generates a deterministic normalized feature vector for text."""
    tokens = tokenize(text)
    vec = np.zeros(dim, dtype=np.float32)
    if not tokens:
        return vec

    for t in tokens:
        vec[hash(t) % dim] += 1.0
        # 3-char subword hashing for typo & morphological tolerance
        if len(t) >= 3:
            for j in range(len(t) - 2):
                vec[hash(t[j : j + 3]) % dim] += 0.5

    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def dense_search(query: str, chunks: List[Dict[str, Any]], dim: int = 128) -> List[Dict[str, Any]]:
    """
    Computes cosine similarity between query and chunks.
    Returns chunks ranked by semantic similarity score.
    """
    if not chunks:
        return []

    q_vec = text_to_vector(query, dim=dim)
    doc_vecs = np.vstack([text_to_vector(c["text"], dim=dim) for c in chunks])
    similarities = np.dot(doc_vecs, q_vec)

    results = []
    for i, c in enumerate(chunks):
        item = dict(c)
        item["dense_score"] = round(float(similarities[i]), 4)
        results.append(item)

    results.sort(key=lambda x: x["dense_score"], reverse=True)
    return results


# ==============================================================================
# 3. Hybrid Search: Reciprocal Rank Fusion (RRF)
# ==============================================================================
def hybrid_search(
    query: str,
    chunks: List[Dict[str, Any]],
    top_k: int = 5,
    rrf_k: int = 60,
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
) -> List[Dict[str, Any]]:
    """
    Combines BM25 and Dense search using Reciprocal Rank Fusion:
        RRF(d) = (sparse_weight / (k + rank_bm25)) + (dense_weight / (k + rank_dense))
    """
    if not chunks:
        return []

    bm25_ranked = bm25_search(query, chunks)
    dense_ranked = dense_search(query, chunks)

    # Compute 1-indexed ranks
    bm25_ranks = {item["chunk_id"]: rank + 1 for rank, item in enumerate(bm25_ranked)}
    dense_ranks = {item["chunk_id"]: rank + 1 for rank, item in enumerate(dense_ranked)}

    chunk_lookup = {c["chunk_id"]: c for c in chunks}
    rrf_scores: Dict[str, float] = {}

    for cid in chunk_lookup:
        b_rank = bm25_ranks.get(cid, len(chunks))
        d_rank = dense_ranks.get(cid, len(chunks))
        score = (sparse_weight / (rrf_k + b_rank)) + (dense_weight / (rrf_k + d_rank))
        rrf_scores[cid] = score

    # Sort by highest RRF score
    sorted_cids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]

    results = []
    for cid in sorted_cids:
        item = dict(chunk_lookup[cid])
        item["rrf_score"] = round(rrf_scores[cid], 5)
        item["bm25_rank"] = bm25_ranks[cid]
        item["dense_rank"] = dense_ranks[cid]
        results.append(item)

    return results
