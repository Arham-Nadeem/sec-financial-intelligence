#Run a labeled retrieval benchmark and summarize ranking quality.
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pandas as pd
import yaml

from sec_intelligence.evaluation.metrics import keyword_hit_rate, ndcg_at_k, recall_at_k, reciprocal_rank
from sec_intelligence.models import RetrievalResult


SearchFunction = Callable[[str, int], list[RetrievalResult]]


def load_questions(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return payload["questions"]


def evaluate(search: SearchFunction, questions: list[dict[str, Any]], k: int = 5) -> pd.DataFrame:
    rows = []
    for item in questions:
        results = search(item["question"], k)
        sections = set(item.get("expected_sections", []))
        keywords = set(item.get("expected_keywords", []))
        rows.append(
            {
                "id": item["id"],
                "question": item["question"],
                f"recall@{k}": recall_at_k(results, sections, k),
                "mrr": reciprocal_rank(results, sections),
                f"ndcg@{k}": ndcg_at_k(results, sections, k),
                f"keyword_hit@{k}": keyword_hit_rate(results, keywords, k),
            }
        )
    return pd.DataFrame(rows)
