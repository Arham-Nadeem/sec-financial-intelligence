#Run labeled retrieval benchmarks across BM25, vector, hybrid, and reranked strategies.
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pandas as pd
import yaml

from sec_intelligence.evaluation.metrics import (
    keyword_hit_rate,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    relevant_result_rate,
)
from sec_intelligence.models import RetrievalResult


SearchFunction = Callable[[str, int], list[RetrievalResult]]


def load_questions(path: str | Path) -> list[dict[str, Any]]:
    """Load benchmark questions from YAML."""
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    questions = payload.get("questions", [])
    if not questions:
        raise ValueError("Benchmark file does not contain any questions.")
    return questions


def evaluate_strategy(
    strategy: str,
    search: SearchFunction,
    questions: list[dict[str, Any]],
    k: int = 5,
) -> pd.DataFrame:
    """Evaluate one retrieval strategy against the same labeled questions."""
    rows: list[dict[str, Any]] = []
    for item in questions:
        results = search(item["question"], k)
        sections = set(item.get("expected_sections", []))
        keywords = set(item.get("expected_keywords", []))
        rows.append(
            {
                "strategy": strategy,
                "id": item["id"],
                "question": item["question"],
                f"recall@{k}": recall_at_k(results, sections, k),
                "mrr": reciprocal_rank(results, sections),
                f"ndcg@{k}": ndcg_at_k(results, sections, k),
                f"relevant_rate@{k}": relevant_result_rate(results, sections, k),
                f"keyword_hit@{k}": keyword_hit_rate(results, keywords, k),
                "top_section": results[0].chunk.section_id if results else None,
                "top_chunk_id": results[0].chunk.chunk_id if results else None,
            }
        )
    return pd.DataFrame(rows)


def summarize_results(results: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """Aggregate per-question benchmark results into one row per strategy."""
    if results.empty:
        return pd.DataFrame()
    metric_columns = [
        f"recall@{k}",
        "mrr",
        f"ndcg@{k}",
        f"relevant_rate@{k}",
        f"keyword_hit@{k}",
    ]
    summary = results.groupby("strategy", as_index=False)[metric_columns].mean()
    return summary.sort_values([f"ndcg@{k}", "mrr"], ascending=False).reset_index(drop=True)
