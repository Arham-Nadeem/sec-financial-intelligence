#Compare BM25, vector, hybrid, and reranked SEC retrieval for one research query.
from __future__ import annotations

import argparse
from pathlib import Path

from sec_intelligence.retrieval.bm25 import BM25Retriever, load_chunks
from sec_intelligence.retrieval.hybrid import HybridRetriever
from sec_intelligence.retrieval.reranker import CrossEncoderReranker
from sec_intelligence.retrieval.vector import load_saved_vector_retriever


def print_results(label: str, results: list, top_k: int) -> None:
    print("\n" + "=" * 96)
    print(label)
    print("=" * 96)
    for rank, result in enumerate(results[:top_k], start=1):
        chunk = result.chunk
        score = result.rerank_score if result.rerank_score is not None else result.score
        print(
            f"{rank:>2}. {score:>9.4f} | {chunk.form} | {chunk.filing_date} | "
            f"{chunk.section_id:<16} | {chunk.chunk_id}"
        )
        print(f"    {' '.join(chunk.text.split())[:170]}...")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare SEC retrieval strategies for one query.")
    parser.add_argument("ticker", help="Stock ticker to search.")
    parser.add_argument("query", help="Research query to compare.")
    parser.add_argument("--top-k", type=int, default=5, help="Results shown for each strategy.")
    parser.add_argument("--candidate-k", type=int, default=20, help="Hybrid candidates before reranking.")
    parser.add_argument("--skip-reranker", action="store_true", help="Skip cross-encoder comparison.")
    args = parser.parse_args()

    ticker = args.ticker.upper()
    root = Path(__file__).resolve().parents[1]
    chunks_path = root / "data" / "processed" / ticker / "chunks.jsonl"
    index_dir = root / "data" / "indexes" / ticker / "vector"

    chunks = load_chunks(chunks_path)
    bm25 = BM25Retriever(chunks)
    vector = load_saved_vector_retriever(chunks_path, index_dir)
    hybrid = HybridRetriever(bm25, vector)

    bm25_results = bm25.search(args.query, top_k=args.top_k)
    vector_results = vector.search(args.query, top_k=args.top_k)
    hybrid_candidates = hybrid.search(
        args.query,
        top_k=max(args.candidate_k, args.top_k),
        candidate_k=args.candidate_k,
    )

    print(f"Ticker: {ticker}")
    print(f"Query: {args.query}")
    print_results("BM25", bm25_results, args.top_k)
    print_results("VECTOR", vector_results, args.top_k)
    print_results("HYBRID WEIGHTED RRF", hybrid_candidates, args.top_k)

    if not args.skip_reranker:
        reranked = CrossEncoderReranker().rerank(args.query, hybrid_candidates, top_k=args.top_k)
        print_results("HYBRID + CROSS-ENCODER RERANKER", reranked, args.top_k)


if __name__ == "__main__":
    main()
