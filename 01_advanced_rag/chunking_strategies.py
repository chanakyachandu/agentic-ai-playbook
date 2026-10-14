"""
Document Chunking Strategies for Enterprise RAG.
100% Function-Based: No classes, pure readable Python functions.
"""

import os
import re
from typing import List, Dict, Any


def chunk_fixed_window(
    text: str, chunk_size: int = 400, overlap: int = 80, source: str = "document"
) -> List[Dict[str, Any]]:
    """
    Splits text into sliding windows of words with overlap.
    Returns a list of chunk dictionaries.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be less than chunk_size")

    words = text.split()
    if not words:
        return []

    chunks = []
    stride = chunk_size - overlap
    chunk_idx = 0

    for i in range(0, len(words), stride):
        chunk_words = words[i : i + chunk_size]
        chunk_text = " ".join(chunk_words)

        chunks.append({
            "chunk_id": f"{source}_fixed_{chunk_idx:03d}",
            "text": chunk_text,
            "source_file": source,
            "strategy": "fixed_window",
            "word_count": len(chunk_words),
        })
        chunk_idx += 1

    return chunks


def chunk_by_sections(
    text: str, source: str = "document", min_len: int = 50
) -> List[Dict[str, Any]]:
    """
    Splits text on document headings (SECTION ..., Q01. ..., Markdown ##).
    Returns a list of semantic chunk dictionaries.
    """
    # Pattern to detect section headings or FAQ numbers
    split_pattern = re.compile(
        r"(?=(?:SECTION\s+\d+:|Q\d+\.|\n##\s+|\n###\s+))", re.IGNORECASE
    )
    raw_sections = split_pattern.split(text)

    chunks = []
    chunk_idx = 0

    for sec in raw_sections:
        clean_text = sec.strip()
        if len(clean_text) < min_len:
            continue

        # Extract title from first line
        first_line = clean_text.split("\n", 1)[0].strip()

        chunks.append({
            "chunk_id": f"{source}_sec_{chunk_idx:03d}",
            "text": clean_text,
            "source_file": source,
            "section_title": first_line,
            "strategy": "header_aware",
            "char_length": len(clean_text),
        })
        chunk_idx += 1

    return chunks


def clean_text(text: str) -> str:
    """
    Cleans raw document text before chunking:
    - Removes decorative ASCII borders (===, ---, ***, ___)
    - Collapses excessive blank lines into clean paragraph spacing
    """
    # Remove lines consisting solely of repeated symbols (=, -, *, _)
    cleaned = re.sub(r"^[=\-\*_]{3,}\s*$", "", text, flags=re.MULTILINE)
    # Collapse 3 or more consecutive newlines into 2
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def load_and_chunk_file(
    file_path: str, strategy: str = "header_aware", chunk_size: int = 300, overlap: int = 50
) -> List[Dict[str, Any]]:
    """Loads a text/markdown file, cleans decorative borders, and chunks it."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    filename = os.path.basename(file_path)
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    # Pre-clean document text to remove decorative symbols and noise
    cleaned_content = clean_text(content)

    if strategy == "header_aware":
        return chunk_by_sections(cleaned_content, source=filename)
    else:
        return chunk_fixed_window(cleaned_content, chunk_size=chunk_size, overlap=overlap, source=filename)
