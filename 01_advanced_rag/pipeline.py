"""
Main Advanced RAG Integration Pipeline.
100% Function-Based Architecture: Zero classes, pure composable Python functions.

Integrates all sub-pipelines:
  - Sub-Pipeline 1: Chunking & Ingestion (chunking_strategies.py)
  - Sub-Pipeline 2: Hybrid Search (hybrid_retriever.py)
  - Sub-Pipeline 3: Re-Ranking & Citation Extraction (reranker.py)
  - Sub-Pipeline 4: Grounded LLM Generation (llm_generator.py)
"""

import os
import sys
import time
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Sub-pipeline imports (supports both relative package and direct module imports)
try:
    from .chunking_strategies import load_and_chunk_file
    from .hybrid_retriever import hybrid_search
    from .reranker import rerank_candidates, extract_citations
    from .llm_generator import generate_llm_answer, generate_grounded_response
except ImportError:
    from chunking_strategies import load_and_chunk_file
    from hybrid_retriever import hybrid_search
    from reranker import rerank_candidates, extract_citations
    from llm_generator import generate_llm_answer, generate_grounded_response

load_dotenv()


def resolve_data_directory(data_dir: str = "data/ecommerce") -> str:
    """Finds data/ecommerce regardless of working directory."""
    if os.path.exists(data_dir):
        return data_dir

    file_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join("..", data_dir),
        os.path.join(file_dir, "..", "data", "ecommerce"),
        os.path.join(file_dir, "data", "ecommerce"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c

    return data_dir


def load_knowledge_base(data_dir: str = "data/ecommerce") -> List[Dict[str, Any]]:
    """Sub-Pipeline 1: Loads and chunks all files into a flat list of chunk dicts."""
    resolved_dir = resolve_data_directory(data_dir)
    if not os.path.exists(resolved_dir):
        raise FileNotFoundError(f"Directory not found: {resolved_dir}")

    all_chunks = []
    for root, _, files in os.walk(resolved_dir):
        for file in sorted(files):
            if file.endswith((".txt", ".md")):
                full_path = os.path.join(root, file)
                file_chunks = load_and_chunk_file(full_path, strategy="header_aware")
                all_chunks.extend(file_chunks)

    if not all_chunks:
        raise ValueError(f"No valid text documents found in: {resolved_dir}")

    return all_chunks


def run_pipeline(
    question: str,
    chunks: Optional[List[Dict[str, Any]]] = None,
    data_dir: str = "data/ecommerce",
    top_k: int = 4,
    rerank_top_n: int = 2,
) -> Dict[str, Any]:
    """
    Main Integrated Pipeline. Takes user question, runs all sub-pipelines, returns grounded answer.
    """
    start_time = time.time()

    # Step 1: Chunking & Ingestion
    if chunks is None:
        chunks = load_knowledge_base(data_dir=data_dir)

    # Step 2: Hybrid Search (BM25 + Vectors merged via RRF)
    retrieved = hybrid_search(question, chunks, top_k=top_k)

    # Step 3: Re-ranking & Citations
    reranked = rerank_candidates(question, retrieved, top_n=rerank_top_n)
    citations = extract_citations(question, reranked)

    # Step 4: Context Assembly & Answer Generation (llm_generator.py)
    answer = generate_grounded_response(question, reranked, citations)
    elapsed_ms = round((time.time() - start_time) * 1000, 2)

    return {
        "query": question,
        "answer": answer,
        "citations": citations,
        "retrieved_chunks": reranked,
        "latency_ms": elapsed_ms,
    }


# Backwards compatibility alias
ask_rag = run_pipeline


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")

    query = "What is the return policy for laptops and high-value electronics?"
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])

    print("Running Master RAG Pipeline...")
    res = run_pipeline(query)
    print(f"\nQuery: {res['query']}")
    print(f"Answer:\n{res['answer']}\n")
    print(f"Latency: {res['latency_ms']} ms")
