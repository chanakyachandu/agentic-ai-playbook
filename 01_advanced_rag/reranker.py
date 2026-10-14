"""
Cross-Encoder Re-Ranking and Citation Grounding.
100% Function-Based: No classes, pure readable Python functions.
"""

import re
from typing import List, Dict, Any


def rerank_candidates(
    query: str, candidates: List[Dict[str, Any]], top_n: int = 3
) -> List[Dict[str, Any]]:
    """
    Re-scores candidates based on lexical coverage and exact phrase match.
    Returns the top_n re-ranked candidates.
    """
    query_words = set(re.findall(r"\b[a-z0-9]+\b", query.lower()))
    if not query_words or not candidates:
        return candidates[:top_n]

    scored = []
    for cand in candidates:
        text = cand["text"].lower()
        matched = sum(1 for w in query_words if w in text)
        coverage = matched / max(len(query_words), 1)

        phrase_bonus = 0.2 if query.lower() in text else 0.0
        base_score = cand.get("rrf_score", 0.0)

        item = dict(cand)
        item["rerank_score"] = round(base_score + (coverage * 0.1) + phrase_bonus, 5)
        item["term_coverage"] = round(coverage, 2)
        scored.append(item)

    scored.sort(key=lambda x: x["rerank_score"], reverse=True)
    return scored[:top_n]


def extract_citations(query: str, top_chunks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Extracts the most relevant verbatim sentence from each chunk to cite as evidence.
    """
    citations = []
    query_terms = [t for t in re.findall(r"\b[a-z0-9]+\b", query.lower()) if len(t) > 2]

    for chunk in top_chunks:
        sentences = re.split(r"(?<=[.!?])\s+", chunk["text"])
        best_sentence = ""
        best_overlap = -1

        for s in sentences:
            clean_s = s.strip()
            if not clean_s:
                continue
            overlap = sum(1 for term in query_terms if term in clean_s.lower())
            if overlap > best_overlap:
                best_overlap = overlap
                best_sentence = clean_s

        if not best_sentence and sentences:
            best_sentence = sentences[0].strip()

        sec_title = chunk.get("section_title") or chunk.get("metadata", {}).get("section_title", "Document Section")
        citations.append({
            "chunk_id": chunk["chunk_id"],
            "source_file": chunk["source_file"],
            "section": sec_title,
            "quote": best_sentence[:180] + ("..." if len(best_sentence) > 180 else ""),
        })

    return citations
