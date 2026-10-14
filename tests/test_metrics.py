#Unit tests for retrieval quality and citation coverage metrics.
from datetime import date

from sec_intelligence.evaluation.metrics import citation_coverage, reciprocal_rank
from sec_intelligence.models import Chunk, RetrievalResult


def test_reciprocal_rank() -> None:
    def result(section: str) -> RetrievalResult:
        chunk = Chunk(
            chunk_id=section,
            ticker="TEST",
            form="10-K",
            filing_date=date(2026, 1, 1),
            accession_number="a",
            section_id=section,
            section_title=section,
            text="text",
            source_url="https://example.com",
        )
        return RetrievalResult(chunk=chunk, score=1.0)

    assert reciprocal_rank([result("item_1"), result("item_7")], {"item_7"}) == 0.5


def test_citation_coverage() -> None:
    answer = "Revenue increased. [10-K 2026 | Item 7 | abc] Risk also changed. [10-K 2026 | Item 1A | def]"
    assert citation_coverage(answer) > 0.0
