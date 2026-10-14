#Unit tests for section-aware chunk creation and stable metadata.
from datetime import date

from sec_intelligence.models import FilingMetadata, FilingSection
from sec_intelligence.parsing.chunker import chunk_sections


def test_chunk_sections_preserves_source_metadata() -> None:
    filing = FilingMetadata(
        ticker="TEST",
        cik="0000000001",
        company_name="Test Company",
        form="10-K",
        accession_number="0000000001-26-000001",
        filing_date=date(2026, 1, 1),
        report_date=date(2025, 12, 31),
        primary_document="test.htm",
        filing_url="https://example.com/test.htm",
    )
    text = " ".join(f"word{i}" for i in range(180))
    sections = [FilingSection(section_id="item_1a", title="Item 1A Risk Factors", text=text)]
    chunks = chunk_sections(filing, sections, chunk_size_words=100, overlap_words=20)

    assert len(chunks) == 2
    assert chunks[0].section_id == "item_1a"
    assert chunks[0].ticker == "TEST"
    assert chunks[0].source_url == filing.filing_url
