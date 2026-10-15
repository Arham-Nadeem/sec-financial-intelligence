#Validate BM25 ranking, filtering, and chunk-file loading behavior.
from __future__ import annotations

from datetime import date
import json

import pytest

from sec_intelligence.models import Chunk
from sec_intelligence.retrieval.bm25 import BM25Retriever, load_chunks


def make_chunk(
    chunk_id: str,
    text: str,
    form: str = "10-K",
    section_id: str = "item_1a",
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        ticker="AAPL",
        form=form,
        filing_date=date(2025, 10, 31),
        accession_number="0000320193-25-000079",
        section_id=section_id,
        section_title="Risk Factors",
        text=text,
        source_url="https://www.sec.gov/example",
    )


def test_bm25_ranks_keyword_relevant_chunk_first() -> None:
    chunks = [
        make_chunk("supply", "Supplier concentration and manufacturing disruptions may affect supply."),
        make_chunk("privacy", "Privacy regulation may increase compliance costs."),
        make_chunk("currency", "Foreign currency fluctuations may affect reported revenue."),
    ]

    results = BM25Retriever(chunks).search("supplier manufacturing supply", top_k=2)

    assert results[0].chunk.chunk_id == "supply"
    assert results[0].bm25_score >= results[1].bm25_score


def test_bm25_supports_form_and_section_filters() -> None:
    chunks = [
        make_chunk("annual", "Artificial intelligence investment increased.", "10-K", "item_7"),
        make_chunk("quarterly", "Artificial intelligence investment increased.", "10-Q", "part_i_item_2"),
    ]

    results = BM25Retriever(chunks).search(
        "artificial intelligence investment",
        top_k=5,
        form="10-Q",
        section_id="part_i_item_2",
    )

    assert [result.chunk.chunk_id for result in results] == ["quarterly"]


def test_load_chunks_reads_jsonl(tmp_path) -> None:
    path = tmp_path / "chunks.jsonl"
    chunk = make_chunk("one", "Revenue increased year over year.")
    path.write_text(json.dumps(chunk.model_dump(mode="json")) + "\n", encoding="utf-8")

    loaded = load_chunks(path)

    assert len(loaded) == 1
    assert loaded[0].chunk_id == "one"


def test_bm25_rejects_empty_query() -> None:
    retriever = BM25Retriever([make_chunk("one", "Revenue increased.")])

    with pytest.raises(ValueError, match="searchable token"):
        retriever.search("---")
