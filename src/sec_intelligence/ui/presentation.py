#Provide lightweight formatting helpers for the Streamlit research interface.
from __future__ import annotations

from collections.abc import Iterable
from difflib import SequenceMatcher

import pandas as pd


METRIC_LABELS = {
    "revenue": "Revenue",
    "operating_income": "Operating income",
    "net_income": "Net income",
    "operating_cash_flow": "Operating cash flow",
    "capex": "Capital expenditures",
    "free_cash_flow": "Free cash flow",
}

EVALUATION_METRIC_LABELS = {
    "recall@5": "Recall@5",
    "mrr": "MRR",
    "ndcg@5": "nDCG@5",
    "relevant_rate@5": "Relevant result rate@5",
    "keyword_hit@5": "Keyword hit rate@5",
}


def format_currency(value: float | int | None) -> str:
    """Format large financial values as compact USD amounts."""
    if value is None or pd.isna(value):
        return "—"
    number = float(value)
    magnitude = abs(number)
    if magnitude >= 1_000_000_000:
        return f"${number / 1_000_000_000:.1f}B"
    if magnitude >= 1_000_000:
        return f"${number / 1_000_000:.1f}M"
    if magnitude >= 1_000:
        return f"${number / 1_000:.1f}K"
    return f"${number:,.0f}"


def format_percent(value: float | int | None) -> str:
    """Format a percentage-point value for KPI deltas."""
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):+.1f}%"


def available_metrics(columns: Iterable[str]) -> list[str]:
    """Return financial metrics in a consistent business-friendly order."""
    column_set = set(columns)
    return [metric for metric in METRIC_LABELS if metric in column_set]


def strategy_display_name(strategy: str) -> str:
    """Convert internal retrieval strategy names into readable UI labels."""
    labels = {
        "bm25": "BM25",
        "vector": "Vector",
        "hybrid": "Hybrid",
        "hybrid_reranked": "Hybrid + reranker",
    }
    return labels.get(strategy, strategy.replace("_", " ").title())


def comparison_similarity_note(excerpts: list[str]) -> str | None:
    """Summarize when selected comparison passages are effectively unchanged."""
    if len(excerpts) < 2:
        return None
    normalized = [" ".join(text.lower().split()) for text in excerpts if text.strip()]
    if len(normalized) < 2:
        return None
    similarities = [
        SequenceMatcher(None, normalized[index - 1], normalized[index]).ratio()
        for index in range(1, len(normalized))
    ]
    if similarities and min(similarities) >= 0.95:
        return "The top matched disclosure language is effectively unchanged across the selected filings."
    return None
