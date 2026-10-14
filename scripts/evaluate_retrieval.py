#Compare lexical, semantic, hybrid, and reranked retrieval on the labeled benchmark.
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from sec_intelligence.config import PROJECT_ROOT, load_settings
from sec_intelligence.evaluation.runner import evaluate, load_questions
from sec_intelligence.pipeline import load_chunks
from sec_intelligence.retrieval.hybrid import HybridRetriever


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ticker")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    settings = load_settings()
    chunks_path = PROJECT_ROOT / "data" / "processed" / args.ticker.upper() / "chunks.jsonl"
    questions_path = PROJECT_ROOT / "data" / "evaluation" / "questions.yaml"
    chunks = load_chunks(chunks_path)
    config = settings["retrieval"]
    retriever = HybridRetriever(
        chunks=chunks,
        embedding_model_name=config["embedding_model"],
        reranker_model_name=config["reranker_model"],
        vector_weight=float(config["vector_weight"]),
        bm25_weight=float(config["bm25_weight"]),
        use_reranker=bool(config["use_reranker"]),
    )
    questions = load_questions(questions_path)
    experiments = [
        ("bm25", "bm25", False),
        ("vector", "vector", False),
        ("hybrid", "hybrid", False),
        ("hybrid_reranked", "hybrid", True),
    ]
    summaries = []
    output_dir = PROJECT_ROOT / "data" / "processed" / args.ticker.upper() / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, mode, rerank in experiments:
        def search(query: str, k: int):
            return retriever.search(query, candidate_k=max(20, k), top_k=k, mode=mode, apply_reranker=rerank)

        frame = evaluate(search, questions, k=args.k)
        frame.to_csv(output_dir / f"{name}.csv", index=False)
        summary = frame.select_dtypes(include="number").mean().to_dict()
        summary["strategy"] = name
        summaries.append(summary)

    summary_frame = pd.DataFrame(summaries).set_index("strategy")
    summary_frame.to_csv(output_dir / "summary.csv")
    print(summary_frame.round(3).to_string())


if __name__ == "__main__":
    main()
