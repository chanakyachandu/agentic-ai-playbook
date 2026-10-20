"""
The RAG Triad Automated Evaluation Engine.
100% Function-Based: Pure Python functions implementing Ragas-style metrics.

Metrics:
1. Faithfulness: Is the generated answer factually grounded in the retrieved context?
2. Answer Relevance: Does the generated response directly answer the user's prompt?
3. Context Precision: Were the most relevant context chunks positioned at the top of retrieval?
"""

import re
from typing import Dict, Any, List, Optional
import numpy as np


def clean_tokens(text: str) -> List[str]:
    """Extracts alphanumeric tokens, filtering conversational and grammatical stopwords."""
    stopwords = {
        "a", "an", "the", "in", "on", "at", "to", "for", "of", "and", "or", "is",
        "are", "was", "were", "be", "this", "that", "it", "with", "as", "by", "from",
        "you", "your", "we", "our", "they", "their", "i", "can", "could", "would",
        "should", "will", "shall", "may", "might", "have", "has", "had", "do", "does",
        "did", "if", "but", "so", "than", "then", "what", "how", "when", "where",
        "who", "which", "there", "their", "subject", "regarding"
    }
    raw = re.findall(r"\b[a-z0-9]+\b", text.lower())
    return [w for w in raw if w not in stopwords and len(w) > 1]


def compute_token_overlap(source_tokens: List[str], target_tokens: List[str]) -> float:
    """Computes precision-style token overlap of source within target with stem tolerance."""
    if not source_tokens:
        return 0.0
    target_set = set(target_tokens)
    matched = 0
    for t in source_tokens:
        if t in target_set:
            matched += 1
        elif len(t) >= 4 and any(t.startswith(tt[:4]) or tt.startswith(t[:4]) for tt in target_tokens if len(tt) >= 4):
            matched += 1
    return matched / len(source_tokens)


def evaluate_faithfulness(answer: str, retrieved_contexts: List[str]) -> float:
    """
    Evaluates factual grounding (anti-hallucination metric).
    Measures what fraction of factual claim tokens in the answer exist in the context.
    Returns: Score between 0.0 and 1.0.
    """
    if not answer or not retrieved_contexts:
        return 0.0

    answer_tokens = clean_tokens(answer)
    if not answer_tokens:
        return 1.0

    context_corpus = " ".join(retrieved_contexts)
    context_tokens = clean_tokens(context_corpus)

    score = compute_token_overlap(answer_tokens, context_tokens)
    return round(float(min(1.0, score)), 4)


def evaluate_answer_relevance(query: str, answer: str) -> float:
    """
    Evaluates whether the answer addresses the intent of the prompt.
    Measures semantic topic overlap between query and generated response.
    Returns: Score between 0.0 and 1.0.
    """
    if not query or not answer:
        return 0.0

    q_tokens = clean_tokens(query)
    a_tokens = clean_tokens(answer)

    if not q_tokens:
        return 1.0

    overlap = compute_token_overlap(q_tokens, a_tokens)
    # Give a baseline boost if answer contains sufficient informative depth
    length_penalty = min(1.0, len(a_tokens) / max(len(q_tokens), 1))
    score = (overlap * 0.7) + (length_penalty * 0.3)
    return round(float(min(1.0, max(0.0, score))), 4)


def evaluate_context_precision(
    ground_truth_keywords: List[str],
    retrieved_contexts: List[str],
) -> float:
    """
    Evaluates retrieval ranking quality (Mean Average Precision style).
    Higher score if top ranked chunks contain target concepts.
    Returns: Score between 0.0 and 1.0.
    """
    if not ground_truth_keywords or not retrieved_contexts:
        return 0.0

    target_set = {k.lower().strip() for k in ground_truth_keywords if k}
    if not target_set:
        return 0.0

    precisions = []
    relevant_found = 0

    for rank, chunk in enumerate(retrieved_contexts, start=1):
        chunk_lower = chunk.lower()
        is_relevant = any(k in chunk_lower for k in target_set)
        if is_relevant:
            relevant_found += 1
            precisions.append(relevant_found / rank)

    if not precisions:
        return 0.0

    score = sum(precisions) / max(len(precisions), 1)
    return round(float(min(1.0, score)), 4)


def evaluate_rag_triad(
    query: str,
    answer: str,
    retrieved_contexts: List[str],
    ground_truth_keywords: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Computes all three RAG Triad scores plus composite geometric mean.
    """
    faithfulness = evaluate_faithfulness(answer, retrieved_contexts)
    relevance = evaluate_answer_relevance(query, answer)

    # If no ground truth keywords are provided, use key query terms
    keywords = ground_truth_keywords if ground_truth_keywords else clean_tokens(query)
    precision = evaluate_context_precision(keywords, retrieved_contexts)

    # Harmonic mean composite score
    triad_scores = [faithfulness, relevance, precision]
    if all(s > 0 for s in triad_scores):
        composite = 3.0 / sum(1.0 / s for s in triad_scores)
    else:
        composite = sum(triad_scores) / 3.0

    passed = faithfulness >= 0.5 and relevance >= 0.5 and precision >= 0.5

    return {
        "faithfulness": faithfulness,
        "answer_relevance": relevance,
        "context_precision": precision,
        "composite_score": round(float(composite), 4),
        "passed": passed,
    }


def run_eval_suite(test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Executes RAG Triad evaluation over a batch of test cases for CI/CD gates.
    """
    results = []
    passed_count = 0

    for item in test_cases:
        res = evaluate_rag_triad(
            query=item.get("query", ""),
            answer=item.get("answer", ""),
            retrieved_contexts=item.get("contexts", []),
            ground_truth_keywords=item.get("ground_truth_keywords"),
        )
        if res["passed"]:
            passed_count += 1
        results.append({"query": item.get("query"), "evaluation": res})

    total = len(test_cases)
    pass_rate = round((passed_count / total) * 100, 2) if total > 0 else 0.0

    return {
        "total_evaluated": total,
        "passed_cases": passed_count,
        "pass_rate_pct": pass_rate,
        "suite_passed": pass_rate >= 80.0,
        "details": results,
    }
