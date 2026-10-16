#Test retrieval evaluation metrics with deterministic filing-section labels.
from __future__ import annotations

from datetime import date

import pytest

from sec_intelligence.evaluation.metrics import (
    keyword_hit_rate,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    relevant_result_rate,
)
from sec_intelligence.models import Chunk, RetrievalResult


def result(chunk_id: str, section_id: str, text: str = "supplier manufacturing risk") -> RetrievalResult:
    return RetrievalResult(
        chunk=Chunk(
            chunk_id=chunk_id,
            ticker="AAPL",
            form="10-K",
            filing_date=date(2025, 10, 31),
            accession_number="0001",
            section_id=section_id,
            section_title=section_id,
            text=text,
            source_url="https://www.sec.gov/example",
        ),
        score=1.0,
    )


def test_ranking_metrics_reward_relevant_sections() -> None:
    results = [result("a", "item_1"), result("b", "item_1a"), result("c", "item_7")]
    expected = {"item_1a", "item_7"}
    assert recall_at_k(results, expected, 3) == 1.0
    assert reciprocal_rank(results, expected) == 0.5
    assert 0.0 <= ndcg_at_k(results, expected, 3) <= 1.0
    assert ndcg_at_k(results, expected, 3) > 0.6
    assert relevant_result_rate(results, expected, 3) == pytest.approx(2 / 3)


def test_ndcg_does_not_double_count_repeated_sections() -> None:
    results = [
        result("a", "item_1a"),
        result("b", "item_1a"),
        result("c", "item_1a"),
        result("d", "item_7"),
    ]
    expected = {"item_1a", "item_7"}
    score = ndcg_at_k(results, expected, 4)
    assert 0.0 <= score <= 1.0
    assert score < 1.0


def test_keyword_hit_rate_tracks_expected_evidence_terms() -> None:
    results = [result("a", "item_1a", "Supplier manufacturing dependency can disrupt supply.")]
    assert keyword_hit_rate(results, {"supplier", "manufacturing", "cash"}, 1) == pytest.approx(2 / 3)
