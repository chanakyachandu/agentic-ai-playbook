"""
Unit Tests for Module 1: Advanced RAG System (100% Function-Based).
Verifies pure functions: Chunking, BM25, Dense Vector, RRF Fusion, and Citation Grounding.
"""

import pytest
import os
import sys
import importlib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import functions dynamically
chunking_mod = importlib.import_module("01_advanced_rag.chunking_strategies")
retriever_mod = importlib.import_module("01_advanced_rag.hybrid_retriever")
reranker_mod = importlib.import_module("01_advanced_rag.reranker")
generator_mod = importlib.import_module("01_advanced_rag.llm_generator")
pipeline_mod = importlib.import_module("01_advanced_rag.pipeline")

chunk_fixed_window = chunking_mod.chunk_fixed_window
chunk_by_sections = chunking_mod.chunk_by_sections
clean_text = chunking_mod.clean_text
bm25_search = retriever_mod.bm25_search
dense_search = retriever_mod.dense_search
hybrid_search = retriever_mod.hybrid_search
rerank_candidates = reranker_mod.rerank_candidates
extract_citations = reranker_mod.extract_citations
format_context_block = generator_mod.format_context_block
generate_grounded_response = generator_mod.generate_grounded_response
run_pipeline = pipeline_mod.run_pipeline
ask_rag = pipeline_mod.ask_rag
load_knowledge_base = pipeline_mod.load_knowledge_base


def test_clean_text():
    dirty = "===\nHello World\n---\n\n\n\nSection Content\n***"
    cleaned = clean_text(dirty)
    assert "===" not in cleaned
    assert "---" not in cleaned
    assert "***" not in cleaned
    assert "Hello World" in cleaned
    assert "Section Content" in cleaned


@pytest.fixture
def sample_chunks():
    return [
        {
            "chunk_id": "chunk_01",
            "text": "SECTION 1: Laptop and computer returns are accepted within 14 calendar days of delivery.",
            "source_file": "policy.txt",
            "section_title": "SECTION 1",
        },
        {
            "chunk_id": "chunk_02",
            "text": "SECTION 2: Standard apparel and accessories may be returned within 30 days for a full refund.",
            "source_file": "policy.txt",
            "section_title": "SECTION 2",
        },
        {
            "chunk_id": "chunk_03",
            "text": "SECTION 3: International shipping rates vary by destination country and customs fees.",
            "source_file": "policy.txt",
            "section_title": "SECTION 3",
        },
    ]


def test_chunk_fixed_window():
    sample_text = "word " * 100
    chunks = chunk_fixed_window(sample_text, chunk_size=30, overlap=10, source="doc.txt")

    assert len(chunks) >= 4
    assert all(c["source_file"] == "doc.txt" for c in chunks)
    assert chunks[0]["chunk_id"] == "doc.txt_fixed_000"


def test_chunk_by_sections():
    text = (
        "SECTION 1: Overview\nThis is general overview text that meets the minimum length requirement.\n\n"
        "SECTION 2: Returns Policy\nItems can be returned within 30 days with original packaging."
    )
    chunks = chunk_by_sections(text, source="policy.txt", min_len=20)

    assert len(chunks) == 2
    assert "SECTION 1" in chunks[0]["section_title"]
    assert "SECTION 2" in chunks[1]["section_title"]


def test_bm25_search(sample_chunks):
    results = bm25_search("laptop computer returns", sample_chunks)

    assert len(results) == 3
    assert results[0]["chunk_id"] == "chunk_01"
    assert results[0]["bm25_score"] > 0.0


def test_dense_search(sample_chunks):
    results = dense_search("international shipping customs", sample_chunks)

    assert len(results) == 3
    assert results[0]["chunk_id"] == "chunk_03"
    assert results[0]["dense_score"] > 0.0


def test_hybrid_search(sample_chunks):
    results = hybrid_search("laptop returns", sample_chunks, top_k=2)

    assert len(results) == 2
    assert results[0]["chunk_id"] == "chunk_01"
    assert "rrf_score" in results[0]
    assert results[0]["rrf_score"] > results[1]["rrf_score"]


def test_rerank_and_citations(sample_chunks):
    candidates = hybrid_search("30 days refund apparel", sample_chunks, top_k=3)
    top_reranked = rerank_candidates("30 days refund apparel", candidates, top_n=1)
    citations = extract_citations("30 days refund apparel", top_reranked)

    assert len(top_reranked) == 1
    assert top_reranked[0]["chunk_id"] == "chunk_02"
    assert len(citations) == 1
    assert "apparel" in citations[0]["quote"].lower()


def test_ask_rag():
    chunks = load_knowledge_base("data/ecommerce")
    result = ask_rag("laptop return window limit", chunks=chunks)

    assert "query" in result
    assert "answer" in result
    assert "citations" in result
    assert len(result["citations"]) > 0
    assert result["latency_ms"] >= 0.0


def test_run_pipeline_single_call():
    # Master pipeline runs end-to-end taking only the query
    result = run_pipeline("laptop return window limit")

    assert "query" in result
    assert "answer" in result
    assert "citations" in result
    assert len(result["citations"]) > 0
    assert result["latency_ms"] >= 0.0


def test_llm_generator(sample_chunks):
    # Step 4: Context formatting & grounded answer generation
    context = format_context_block(sample_chunks[:2])
    assert "[1] Source:" in context
    assert "[2] Source:" in context

    citations = [{"quote": "Return within 14 days", "source_file": "policy.txt", "section": "Returns"}]
    ans = generate_grounded_response("return window", sample_chunks[:2], citations)
    assert len(ans) > 0
    assert "verified" in ans.lower() or "return" in ans.lower()


