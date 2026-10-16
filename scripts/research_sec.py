#Run grounded SEC research over hybrid retrieval with optional filing-to-filing comparison.
from __future__ import annotations

import argparse
from pathlib import Path

from sec_intelligence.answering.grounded import build_grounded_answer, compare_filings, validate_citations
from sec_intelligence.retrieval.bm25 import BM25Retriever, load_chunks
from sec_intelligence.retrieval.hybrid import HybridRetriever
from sec_intelligence.retrieval.reranker import CrossEncoderReranker
from sec_intelligence.retrieval.vector import load_saved_vector_retriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Research SEC filings with grounded source citations.")
    parser.add_argument("ticker", help="Stock ticker to research.")
    parser.add_argument("query", help="Natural-language research question.")
    parser.add_argument("--top-k", type=int, default=4, help="Number of cited evidence passages.")
    parser.add_argument("--candidate-k", type=int, default=20, help="Hybrid candidates before reranking.")
    parser.add_argument("--form", help="Optional filing form filter such as 10-K or 10-Q.")
    parser.add_argument("--section", help="Optional normalized section filter such as item_1a.")
    parser.add_argument("--compare", action="store_true", help="Prefer evidence from distinct filing dates.")
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
    reranked = CrossEncoderReranker().rerank(
        args.query,
        candidates,
        top_k=max(args.candidate_k, args.top_k),
    )

    if args.compare:
        grounded = compare_filings(args.query, reranked, max_filings=args.top_k)
        mode = "Filing comparison"
    else:
        grounded = build_grounded_answer(args.query, reranked, max_evidence=args.top_k)
        mode = "Grounded research"

    validation = validate_citations(grounded)
    print(f"Ticker: {ticker}")
    print(f"Mode: {mode}")
    print(f"Question: {args.query}")
    print(f"Evidence passages: {grounded.evidence_count}")
    print(f"Citation precision: {validation['citation_precision']:.3f}")
    print(f"Citation coverage: {validation['citation_coverage']:.3f}")
    print("\nANSWER")
    print("-" * 88)
    print(grounded.answer)
    print("\nSOURCES")
    print("-" * 88)
    for citation in grounded.citations:
        print(
            f"[{citation.citation_id}] {citation.ticker} {citation.form} | "
            f"{citation.filing_date} | {citation.section_title}"
        )
        print(f"    {citation.source_url}")
        print(f"    {citation.excerpt}")


if __name__ == "__main__":
    main()
