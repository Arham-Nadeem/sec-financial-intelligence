#Benchmark BM25, vector, hybrid, and cross-encoder-reranked SEC retrieval strategies.
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from sec_intelligence.evaluation.runner import evaluate_strategy, load_questions, summarize_results
from sec_intelligence.retrieval.bm25 import BM25Retriever, load_chunks
from sec_intelligence.retrieval.hybrid import HybridRetriever
from sec_intelligence.retrieval.reranker import CrossEncoderReranker
from sec_intelligence.retrieval.vector import load_saved_vector_retriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SEC filing retrieval strategies.")
    parser.add_argument("ticker", help="Stock ticker with parsed chunks and a vector index.")
    parser.add_argument("--k", type=int, default=5, help="Top-k cutoff used for evaluation metrics.")
    parser.add_argument("--candidate-k", type=int, default=20, help="Candidates kept before reranking.")
    parser.add_argument(
        "--questions",
        help="Optional benchmark YAML path. Defaults to data/evaluation/<ticker>_questions.yaml.",
    )
    parser.add_argument("--skip-reranker", action="store_true", help="Skip cross-encoder evaluation.")
    args = parser.parse_args()

    if args.k <= 0 or args.candidate_k <= 0:
        raise ValueError("k and candidate-k must be positive.")

    ticker = args.ticker.upper()
    root = Path(__file__).resolve().parents[1]
    chunks_path = root / "data" / "processed" / ticker / "chunks.jsonl"
    index_dir = root / "data" / "indexes" / ticker / "vector"
    default_questions = root / "data" / "evaluation" / f"{ticker.lower()}_questions.yaml"
    fallback_questions = root / "data" / "evaluation" / "questions.yaml"
    questions_path = Path(args.questions) if args.questions else (
        default_questions if default_questions.exists() else fallback_questions
    )

    chunks = load_chunks(chunks_path)
    bm25 = BM25Retriever(chunks)
    vector = load_saved_vector_retriever(chunks_path, index_dir)
    hybrid = HybridRetriever(bm25, vector)
    questions = load_questions(questions_path)

    frames: list[pd.DataFrame] = []
    frames.append(
        evaluate_strategy(
            "bm25",
            lambda query, k: bm25.search(query, top_k=k),
            questions,
            args.k,
        )
    )
    frames.append(
        evaluate_strategy(
            "vector",
            lambda query, k: vector.search(query, top_k=k),
            questions,
            args.k,
        )
    )
    frames.append(
        evaluate_strategy(
            "hybrid",
            lambda query, k: hybrid.search(
                query,
                top_k=k,
                candidate_k=max(args.candidate_k, k),
            ),
            questions,
            args.k,
        )
    )

    if not args.skip_reranker:
        reranker = CrossEncoderReranker()

        def search_reranked(query: str, k: int):
            candidates = hybrid.search(
                query,
                top_k=max(args.candidate_k, k),
                candidate_k=max(args.candidate_k, k),
            )
            return reranker.rerank(query, candidates, top_k=k)

        frames.append(evaluate_strategy("hybrid_reranked", search_reranked, questions, args.k))

    results = pd.concat(frames, ignore_index=True)
    summary = summarize_results(results, args.k)
    output_dir = root / "data" / "processed" / ticker / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    details_path = output_dir / "retrieval_details.csv"
    summary_path = output_dir / "retrieval_summary.csv"
    results.to_csv(details_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(f"Ticker: {ticker}")
    print(f"Benchmark questions: {len(questions)}")
    print(f"Top-k: {args.k}")
    print(f"Benchmark file: {questions_path}")
    print("\nRetrieval summary:")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nDetailed results: {details_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
