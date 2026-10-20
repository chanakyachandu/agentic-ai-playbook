"""
Retrieval Worker (The Researcher).
100% Function-Based: Zero classes, pure composable Python functions.

Role: Searches store policies, FAQs, and warranty documentation.
Extracts verbatim clauses, return windows, exception rules, and citations.
"""

import os
import re
from typing import Dict, Any, List, Optional


def resolve_data_directory(data_dir: str = "data/ecommerce") -> str:
    """Finds data/ecommerce directory across current and parent folders."""
    if os.path.exists(data_dir):
        return data_dir

    file_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join("..", data_dir),
        os.path.join(file_dir, "..", "data", "ecommerce"),
        os.path.join(file_dir, "data", "ecommerce"),
        os.path.join(os.path.dirname(file_dir), "data", "ecommerce"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return data_dir


def search_policies(
    query: str,
    top_k: int = 3,
    data_dir: str = "data/ecommerce"
) -> List[Dict[str, Any]]:
    """
    Retrieves the most relevant policy and FAQ sections matching the query.
    Attempts to use 01_advanced_rag hybrid retriever if available, otherwise falls back
    to robust keyword and section-aware scoring.
    """
    resolved_dir = resolve_data_directory(data_dir)

    # Attempt to use 01_advanced_rag if available
    try:
        import importlib
        rag_pipe = importlib.import_module("01_advanced_rag.pipeline")
        rag_hybrid = importlib.import_module("01_advanced_rag.hybrid_retriever")
        rag_rerank = importlib.import_module("01_advanced_rag.reranker")

        load_knowledge_base = rag_pipe.load_knowledge_base
        hybrid_search = rag_hybrid.hybrid_search
        rerank_candidates = rag_rerank.rerank_candidates

        chunks = load_knowledge_base(data_dir=resolved_dir)
        candidates = hybrid_search(query, chunks, top_k=top_k * 2)
        reranked = rerank_candidates(query, candidates, top_k=top_k)
        if reranked:
            return [
                {
                    "source": r.get("source", "2026_current_policy.txt"),
                    "section": r.get("section", "Policy"),
                    "content": r.get("content", ""),
                    "score": r.get("rerank_score", r.get("rrf_score", 1.0)),
                }
                for r in reranked
            ]
    except Exception:
        pass

    # Robust built-in policy search fallback
    policy_file = os.path.join(resolved_dir, "2026_current_policy.txt")
    faq_file = os.path.join(resolved_dir, "store_faq_kb.txt")
    target_files = [f for f in [policy_file, faq_file] if os.path.exists(f)]

    sections: List[Dict[str, Any]] = []
    for filepath in target_files:
        filename = os.path.basename(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        raw_parts = re.split(r"\n(?=SECTION\s+\d+|---|\bFAQ\s+\d+)", text)
        for part in raw_parts:
            clean = part.strip()
            if len(clean) > 50:
                header = clean.split("\n")[0][:80].strip("- =")
                sections.append({
                    "source": filename,
                    "section": header,
                    "content": clean,
                })

    tokens = [t.lower() for t in re.findall(r"\w+", query) if len(t) > 2]
    scored = []
    for sec in sections:
        content_lower = sec["content"].lower()
        score = sum(content_lower.count(tok) for tok in tokens)
        if score > 0:
            scored.append({**sec, "score": float(score)})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k] if scored else sections[:top_k]


def extract_policy_citations(query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extracts verifiable citations and specific clauses from retrieved docs."""
    citations = []
    for doc in documents:
        content = doc.get("content", "")
        # Match section markers like Section 1.2 or 3.2
        section_matches = re.findall(r"(?:Section\s+)?(\d+\.\d+[^:\n]*:[^\n]+)", content, re.IGNORECASE)
        if section_matches:
            for sm in section_matches[:2]:
                citations.append({
                    "source": doc.get("source", "2026_current_policy.txt"),
                    "clause": sm.strip(),
                    "quote": sm.strip()[:140],
                })
        else:
            meaningful_lines = [l.strip("- =# ") for l in content.split("\n") if l.strip("- =# ")]
            first_line = meaningful_lines[0] if meaningful_lines else doc.get("section", "Policy Clause")
            citations.append({
                "source": doc.get("source", "2026_current_policy.txt"),
                "clause": first_line[:80],
                "quote": content[:140].replace("\n", " "),
            })
    return citations


def run_retrieval_worker(
    query: str,
    category: Optional[str] = None,
    data_dir: str = "data/ecommerce"
) -> Dict[str, Any]:
    """
    Main entry point for the Retrieval Worker (Researcher).
    Searches relevant policies based on query and optional category filter.
    """
    search_query = f"{category} {query}" if category else query
    docs = search_policies(search_query, top_k=3, data_dir=data_dir)
    citations = extract_policy_citations(query, docs)

    summary_bullets = []
    for doc in docs[:2]:
        first_meaningful = [
            line.strip() for line in doc["content"].split("\n")
            if line.strip() and not line.strip().startswith(("=", "-", "#"))
        ]
        if first_meaningful:
            summary_bullets.append(first_meaningful[0][:150])

    return {
        "worker": "Retrieval Worker (Researcher)",
        "status": "SUCCESS" if docs else "NO_RESULTS",
        "search_query": search_query,
        "document_count": len(docs),
        "documents": docs,
        "citations": citations,
        "summary": " | ".join(summary_bullets) if summary_bullets else "Policy rules retrieved.",
    }


def deep_research_policy(
    topic: str,
    category: Optional[str] = None,
    vip_tier: Optional[str] = None,
    data_dir: str = "data/ecommerce"
) -> Dict[str, Any]:
    """
    Multi-Agent Deep Researcher Pattern.
    Conducts iterative multi-angle policy research by generating targeted sub-queries,
    aggregating distinct policy sections, and cross-verifying findings.
    """
    # 1. Generate targeted sub-queries for deep investigation
    sub_queries = [
        f"{category or ''} return window days and device runtime limits",
        f"{category or ''} restocking fee packaging manufacturer seal intact",
        f"NovaClub VIP {vip_tier or 'Silver'} loyalty tier return window restocking fee waiver",
        f"{topic} hardware warranty coverage repair exchange",
    ]

    all_docs: List[Dict[str, Any]] = []
    seen_contents = set()

    for sq in sub_queries:
        docs = search_policies(sq.strip(), top_k=2, data_dir=data_dir)
        for d in docs:
            snippet = d.get("content", "")[:100]
            if snippet not in seen_contents:
                seen_contents.add(snippet)
                all_docs.append(d)

    # 2. Extract and deduplicate citations
    all_citations = extract_policy_citations(topic, all_docs)
    unique_citations = []
    seen_clauses = set()
    for c in all_citations:
        if c["clause"] not in seen_clauses:
            seen_clauses.add(c["clause"])
            unique_citations.append(c)

    # 3. Compile structured research brief
    key_findings = []
    for doc in all_docs[:4]:
        for line in doc["content"].split("\n"):
            clean_l = line.strip()
            if any(term in clean_l.lower() for term in ["return window", "restocking fee", "runtime", "warranty", "vip"]):
                if len(clean_l) > 25 and clean_l not in key_findings:
                    key_findings.append(clean_l[:140])
                    break

    return {
        "worker": "Retrieval Worker (Deep Researcher)",
        "topic": topic,
        "sub_queries_executed": sub_queries,
        "total_documents_analyzed": len(all_docs),
        "verified_citations": unique_citations[:5],
        "key_findings": key_findings[:4],
        "research_brief": " | ".join(key_findings[:3]) if key_findings else "Deep research completed.",
    }


if __name__ == "__main__":
    sample = run_retrieval_worker("MacBook Pro return window and restocking fee", category="Laptops")
    print(f"Worker: {sample['worker']}")
    print(f"Status: {sample['status']}")
    print(f"Citations Found: {len(sample['citations'])}")
    for c in sample["citations"]:
        print(f" - [{c['source']}] {c['clause']}")
