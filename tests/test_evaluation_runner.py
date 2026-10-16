#Test benchmark execution and strategy-level metric aggregation.
from __future__ import annotations

from datetime import date

from sec_intelligence.evaluation.runner import evaluate_strategy, summarize_results
from sec_intelligence.models import Chunk, RetrievalResult


def make_result(section_id: str) -> RetrievalResult:
    return RetrievalResult(
        chunk=Chunk(
            chunk_id=section_id,
            ticker="AAPL",
            form="10-K",
            filing_date=date(2025, 10, 31),
            accession_number="0001",
            section_id=section_id,
            section_title=section_id,
            text="supplier manufacturing risk",
            source_url="https://www.sec.gov/example",
        ),
        score=1.0,
    )


def test_evaluate_strategy_and_summary() -> None:
    questions = [
        {
            "id": "risk",
            "question": "supplier risk",
            "expected_sections": ["item_1a"],
            "expected_keywords": ["supplier"],
        }
    ]

    frame = evaluate_strategy(
        "test",
        lambda query, k: [make_result("item_1a")],
        questions,
        k=5,
    )
    summary = summarize_results(frame, k=5)

    assert frame.iloc[0]["mrr"] == 1.0
    assert summary.iloc[0]["strategy"] == "test"
    assert summary.iloc[0]["recall@5"] == 1.0
