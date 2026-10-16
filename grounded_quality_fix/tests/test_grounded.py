#Test grounded answer creation, relevance filtering, comparison behavior, and citation validation.
from datetime import date

from sec_intelligence.answering.grounded import build_grounded_answer, compare_filings, validate_citations
from sec_intelligence.models import Chunk, RetrievalResult


def _result(
    chunk_id: str,
    filing_date: date,
    text: str,
    score: float = 1.0,
    section_id: str = "item_1a",
) -> RetrievalResult:
    chunk = Chunk(
        chunk_id=chunk_id,
        ticker="AAPL",
        form="10-K",
        filing_date=filing_date,
        accession_number=f"acc-{chunk_id}",
        section_id=section_id,
        section_title="Item 1A - Risk Factors" if section_id == "item_1a" else "Item 2 - Other Information",
        text=text,
        source_url=f"https://example.com/{chunk_id}",
    )
    return RetrievalResult(chunk=chunk, score=score, rerank_score=score)


def test_grounded_answer_creates_source_markers() -> None:
    results = [
        _result(
            "one",
            date(2025, 10, 31),
            "The Company depends on manufacturing services provided by outsourcing partners. Other risks also exist.",
            2.8,
        ),
        _result(
            "two",
            date(2024, 11, 1),
            "Suppliers and contract manufacturers operate across a global supply chain.",
            2.0,
        ),
    ]

    answer = build_grounded_answer("dependence on manufacturing partners", results)

    assert answer.evidence_count == 2
    assert "[1]" in answer.answer
    assert "[2]" in answer.answer
    assert answer.citation_coverage == 1.0


def test_grounded_answer_drops_clearly_weak_tail_results() -> None:
    results = [
        _result("strong", date(2025, 10, 31), "The Company depends on outsourcing partners for manufacturing.", 2.8),
        _result("weak", date(2024, 11, 1), "The Company protects patents and trademarks globally.", -0.6),
    ]

    answer = build_grounded_answer("outside companies for manufacturing", results, max_evidence=4)

    assert answer.evidence_count == 1
    assert "outsourcing partners" in answer.answer
    assert "patents" not in answer.answer


def test_compare_filings_uses_distinct_dates() -> None:
    results = [
        _result("new-a", date(2025, 10, 31), "Outsourcing partners provide manufacturing services.", 3.0),
        _result("new-b", date(2025, 10, 31), "Suppliers support product assembly.", 2.0),
        _result("old", date(2024, 11, 1), "The Company relies on outsourcing partners for final assembly.", 1.5),
    ]

    answer = compare_filings("outsourcing manufacturing risk", results, max_filings=2)

    assert answer.evidence_count == 2
    assert {citation.filing_date for citation in answer.citations} == {
        date(2025, 10, 31),
        date(2024, 11, 1),
    }
    assert answer.answer.startswith("2024-11-01:")


def test_compare_filings_prefers_risk_sections_for_risk_questions() -> None:
    results = [
        _result(
            "wrong-section",
            date(2026, 7, 31),
            "Purchase obligations include supplier arrangements related to manufacturing.",
            3.5,
            section_id="part_ii_item_2",
        ),
        _result(
            "risk-section",
            date(2025, 10, 31),
            "The Company depends on outsourcing partners for manufacturing and logistics.",
            2.7,
            section_id="item_1a",
        ),
    ]

    answer = compare_filings("manufacturing and supplier risk over time", results, max_filings=2)

    assert answer.evidence_count == 1
    assert answer.citations[0].section_id == "item_1a"
    assert "Purchase obligations" not in answer.answer


def test_validate_citations_reports_full_validity() -> None:
    answer = build_grounded_answer(
        "supplier risk",
        [_result("one", date(2025, 10, 31), "Supplier disruptions can affect product availability.", 2.0)],
    )

    validation = validate_citations(answer)

    assert validation["citation_precision"] == 1.0
    assert validation["citation_coverage"] == 1.0
    assert validation["all_citations_valid"] is True
