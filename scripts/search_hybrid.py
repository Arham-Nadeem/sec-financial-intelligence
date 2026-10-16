#Search SEC filings with hybrid BM25/vector retrieval and optional cross-encoder reranking.
from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

from sec_intelligence.retrieval.bm25 import BM25Retriever, load_chunks
from sec_intelligence.retrieval.hybrid import HybridRetriever
from sec_intelligence.retrieval.reranker import CrossEncoderReranker
from sec_intelligence.retrieval.vector import load_saved_vector_retriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Search SEC filings with hybrid retrieval.")
    parser.add_argument("ticker", help="Stock ticker to search.")
    parser.add_argument("query", help="Natural-language research query.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of final results to return.")
    parser.add_argument("--candidate-k", type=int, default=20, help="Candidates kept before reranking.")
    parser.add_argument("--form", help="Optional filing form filter, such as 10-K or 10-Q.")
    parser.add_argument("--section", help="Optional normalized section filter, such as item_1a.")
    parser.add_argument("--no-rerank", action="store_true", help="Return hybrid results without reranking.")
    args = parser.parse_args()

    ticker = args.ticker.upper()
    root = Path(__file__).resolve().parents[1]
    chunks_path = root / "data" / "processed" / ticker / "chunks.jsonl"
    index_dir = root / "data" / "indexes" / ticker / "vector"

    chunks = load_chunks(chunks_path)
    bm25 = BM25Retriever(chunks)
    vector = load_saved_vector_retriever(chunks_path, index_dir)
    hybrid = HybridRetriever(bm25, vector)
    candidates = hybrid.search(
        args.query,
        top_k=max(args.candidate_k, args.top_k),
        candidate_k=args.candidate_k,
        form=args.form,
        section_id=args.section,
    )

    if args.no_rerank:
        results = candidates[: args.top_k]
        mode = "Hybrid weighted RRF"
    else:
        results = CrossEncoderReranker().rerank(args.query, candidates, top_k=args.top_k)
        mode = "Hybrid weighted RRF + cross-encoder reranking"

    print(f"Ticker: {ticker}")
    print(f"Query: {args.query}")
    print(f"Mode: {mode}")
    print(f"Indexed chunks: {len(chunks)}")
    print(f"Results returned: {len(results)}")

    for rank, result in enumerate(results, start=1):
        chunk = result.chunk
        excerpt = textwrap.shorten(" ".join(chunk.text.split()), width=560, placeholder="...")
        rerank = f" | RERANK {result.rerank_score:.4f}" if result.rerank_score is not None else ""
        print("\n" + "=" * 88)
        print(
            f"#{rank} | HYBRID {result.score:.6f}{rerank} | "
            f"{chunk.form} | {chunk.filing_date}"
        )
        print(f"BM25 raw: {result.bm25_score:.4f} | Vector raw: {result.vector_score:.4f}")
        print(f"Section: {chunk.section_title} ({chunk.section_id})")
        print(f"Chunk ID: {chunk.chunk_id}")
        print(f"Source: {chunk.source_url}")
        print("Excerpt:")
        print(textwrap.fill(excerpt, width=88))


if __name__ == "__main__":
    main()
