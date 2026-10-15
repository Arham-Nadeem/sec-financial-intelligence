#Search parsed SEC filing chunks with local semantic embeddings and a persisted FAISS index.
from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

from sec_intelligence.retrieval.vector import load_saved_vector_retriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Search SEC filings with semantic vector retrieval.")
    parser.add_argument("ticker", help="Stock ticker to search.")
    parser.add_argument("query", help="Natural-language research question or search query.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return.")
    parser.add_argument("--form", help="Optional filing form filter, such as 10-K or 10-Q.")
    parser.add_argument("--section", help="Optional normalized section filter, such as item_1a.")
    args = parser.parse_args()

    ticker = args.ticker.upper()
    root = Path(__file__).resolve().parents[1]
    chunks_path = root / "data" / "processed" / ticker / "chunks.jsonl"
    index_dir = root / "data" / "indexes" / ticker / "vector"

    retriever = load_saved_vector_retriever(chunks_path, index_dir)
    results = retriever.search(
        args.query,
        top_k=args.top_k,
        form=args.form,
        section_id=args.section,
    )

    print(f"Ticker: {ticker}")
    print(f"Query: {args.query}")
    print(f"Indexed chunks: {len(retriever.chunks)}")
    print(f"Results returned: {len(results)}")

    for rank, result in enumerate(results, start=1):
        chunk = result.chunk
        excerpt = textwrap.shorten(" ".join(chunk.text.split()), width=560, placeholder="...")
        print("\n" + "=" * 88)
        print(f"#{rank} | VECTOR {result.vector_score:.4f} | {chunk.form} | {chunk.filing_date}")
        print(f"Section: {chunk.section_title} ({chunk.section_id})")
        print(f"Chunk ID: {chunk.chunk_id}")
        print(f"Source: {chunk.source_url}")
        print("Excerpt:")
        print(textwrap.fill(excerpt, width=88))


if __name__ == "__main__":
    main()
