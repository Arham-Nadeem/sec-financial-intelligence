#Unit tests for section-aware chunk creation and stable metadata.
from datetime import date

import pytest

from sec_intelligence.models import FilingMetadata, FilingSection
from sec_intelligence.parsing.chunker import chunk_sections


def _filing() -> FilingMetadata:
    return FilingMetadata(
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


def test_chunk_sections_preserves_source_metadata() -> None:
    text = " ".join(
        f"Sentence {index} contains useful financial research context."
        for index in range(70)
    )
    sections = [FilingSection(section_id="item_1a", title="Item 1A - Risk Factors", text=text)]
    chunks = chunk_sections(_filing(), sections, chunk_size_words=100, overlap_words=20)

    assert len(chunks) >= 3
    assert chunks[0].section_id == "item_1a"
    assert chunks[0].ticker == "TEST"
    assert chunks[0].source_url == _filing().filing_url
    assert chunks[0].metadata["chunk_index"] == 0
    assert chunks[0].metadata["word_count"] > 0


def test_chunk_ids_are_stable_for_same_input() -> None:
    text = " ".join(f"Sentence {index} describes company performance." for index in range(50))
    sections = [FilingSection(section_id="item_7", title="Item 7 - MD&A", text=text)]

    first = chunk_sections(_filing(), sections, chunk_size_words=90, overlap_words=15)
    second = chunk_sections(_filing(), sections, chunk_size_words=90, overlap_words=15)

    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]


def test_chunk_sections_rejects_invalid_overlap() -> None:
    sections = [FilingSection(section_id="item_1", title="Item 1", text="word " * 100)]
    with pytest.raises(ValueError):
        chunk_sections(_filing(), sections, chunk_size_words=50, overlap_words=50)
