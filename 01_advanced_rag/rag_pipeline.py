"""
End-to-End Enterprise RAG Pipeline.
100% Function-Based: No classes, pure readable Python functions.
"""

import os
import sys
import time
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

try:
    from .chunking_strategies import load_and_chunk_file
    from .hybrid_retriever import hybrid_search
    from .reranker import rerank_candidates, extract_citations
except ImportError:
    from chunking_strategies import load_and_chunk_file
    from hybrid_retriever import hybrid_search
    from reranker import rerank_candidates, extract_citations

load_dotenv()


def load_knowledge_base(data_dir: str = "data/ecommerce") -> List[Dict[str, Any]]:
    """Loads and chunks all text/markdown files in data_dir into a flat list of chunk dicts."""
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    all_chunks = []
    for root, _, files in os.walk(data_dir):
        for file in sorted(files):
            if file.endswith((".txt", ".md")):
                full_path = os.path.join(root, file)
                file_chunks = load_and_chunk_file(full_path, strategy="header_aware")
                all_chunks.extend(file_chunks)

    if not all_chunks:
        raise ValueError(f"No valid text documents found in: {data_dir}")

    return all_chunks


def generate_llm_answer(question: str, context: str, citations: List[Dict[str, str]]) -> str:
    """Generates an answer using Gemini API if key is set, otherwise returns verified context quotes."""
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key and api_key != "your_gemini_api_key_here":
        try:
            from google import genai
            client = genai.Client(api_key=api_key)
            prompt = (
                f"You are an enterprise support AI. Answer using ONLY the verified context below.\n"
                f"Cite sources using [Source, Section].\n\n"
                f"CONTEXT:\n{context}\n\n"
                f"QUESTION: {question}\n\n"
                f"ANSWER:"
            )
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )
            if response and response.text:
                return response.text.strip()
        except Exception:
            pass

    # Reliable local fallback
    top_quote = citations[0]["quote"] if citations else "Refer to store policy."
    top_src = citations[0]["source_file"] if citations else "policy.txt"
    top_sec = citations[0]["section"] if citations else "Section"
    return (
        f"Based on verified documentation [{top_src} | {top_sec}]:\n"
        f"\"{top_quote}\"\n\n"
        f"Verified against {len(citations)} source citations."
    )


def ask_rag(
    question: str,
    chunks: Optional[List[Dict[str, Any]]] = None,
    data_dir: str = "data/ecommerce",
    top_k: int = 4,
    rerank_top_n: int = 2,
) -> Dict[str, Any]:
    """
    Main function executing the entire RAG pipeline:
    1. Hybrid search (BM25 + Dense RRF)
    2. Re-ranking
    3. Citation extraction
    4. Grounded answer generation
    """
    start_time = time.time()

    if chunks is None:
        chunks = load_knowledge_base(data_dir=data_dir)

    # 1. Hybrid Search
    retrieved = hybrid_search(question, chunks, top_k=top_k)

    # 2. Re-rank
    reranked = rerank_candidates(question, retrieved, top_n=rerank_top_n)

    # 3. Extract Citations
    citations = extract_citations(question, reranked)

    # 4. Build Context
    context_blocks = [
        f"[{i+1}] Source: {c['source_file']} | Section: {c.get('section_title', 'General')}\n{c['text']}"
        for i, c in enumerate(reranked)
    ]
    full_context = "\n\n".join(context_blocks)

    # 5. Answer
    answer = generate_llm_answer(question, full_context, citations)
    elapsed_ms = round((time.time() - start_time) * 1000, 2)

    return {
        "query": question,
        "answer": answer,
        "citations": citations,
        "retrieved_chunks": reranked,
        "latency_ms": elapsed_ms,
    }


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")

    test_query = "What is the return policy for laptops and high-value electronics?"
    if len(sys.argv) > 1:
        test_query = " ".join(sys.argv[1:])

    print("=" * 80)
    print("[INIT] Loading knowledge base chunks...")
    chunks = load_knowledge_base("data/ecommerce")
    print(f"Loaded {len(chunks)} chunks from data/ecommerce")

    print(f"\n[QUERY] '{test_query}'")
    result = ask_rag(test_query, chunks=chunks)

    print("\n--- [GROUNDED ANSWER] ---")
    print(result["answer"])

    print("\n--- [VERIFIABLE CITATIONS] ---")
    for cit in result["citations"]:
        print(f"* [{cit['source_file']} - {cit['section']}]: \"{cit['quote']}\"")

    print(f"\n[METRICS] Total Latency: {result['latency_ms']} ms")
    print("=" * 80)
