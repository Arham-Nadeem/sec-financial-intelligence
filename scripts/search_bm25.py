#Search parsed SEC filing chunks with the BM25 retrieval baseline.
from __future__ import annotations

import argparse
import textwrap

from sec_intelligence.config import PROJECT_ROOT
from sec_intelligence.retrieval.bm25 import BM25Retriever, load_chunks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search parsed SEC filing chunks with BM25 keyword retrieval."
    )
    parser.add_argument("ticker", help="Company ticker, for example AAPL")
    parser.add_argument("query", help="Research query to search for")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")
    parser.add_argument("--form", choices=["10-K", "10-Q"], help="Optional filing-form filter")
    parser.add_argument("--section", help="Optional section ID filter, for example item_1a")
    parser.add_argument(
        "--excerpt-chars",
        type=int,
        default=500,
        help="Maximum number of characters shown per result",
    )
    args = parser.parse_args()

    ticker = args.ticker.upper()
    chunk_path = PROJECT_ROOT / "data" / "processed" / ticker / "chunks.jsonl"
    chunks = load_chunks(chunk_path)
    retriever = BM25Retriever(chunks)
    results = retriever.search(
        query=args.query,
        top_k=args.top_k,
        form=args.form,
        section_id=args.section,
    )

    print(f"Ticker: {ticker}")
    print(f"Query: {args.query}")
    print(f"Indexed chunks: {len(chunks)}")
    print(f"Results returned: {len(results)}")
    if args.form:
        print(f"Form filter: {args.form}")
    if args.section:
        print(f"Section filter: {args.section}")

    if not results:
        print("No matching chunks found for the selected filters.")
        return

    for rank, result in enumerate(results, start=1):
        chunk = result.chunk
        excerpt = " ".join(chunk.text.split())[: args.excerpt_chars]
        if len(chunk.text) > args.excerpt_chars:
            excerpt += "..."

        print("\n" + "=" * 88)
        print(
            f"#{rank} | BM25 {result.bm25_score:.4f} | "
            f"{chunk.form} | {chunk.filing_date.isoformat()}"
        )
        print(f"Section: {chunk.section_title} ({chunk.section_id})")
        print(f"Chunk ID: {chunk.chunk_id}")
        print(f"Source: {chunk.source_url}")
        print("Excerpt:")
        print(textwrap.fill(excerpt, width=100))


if __name__ == "__main__":
    main()
