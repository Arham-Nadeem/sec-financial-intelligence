#Test semantic retrieval logic without downloading an external embedding model.
from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from sec_intelligence.models import Chunk
from sec_intelligence.retrieval.vector import VectorRetriever, _normalize_rows


class FakeEncoder:
    """Return deterministic embeddings for a small controlled vocabulary."""

    def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
        rows = []
        for text in texts:
            lowered = text.lower()
            if "supplier" in lowered or "manufactur" in lowered or "production" in lowered:
                rows.append([1.0, 0.0, 0.0])
            elif "revenue" in lowered or "sales" in lowered:
                rows.append([0.0, 1.0, 0.0])
            else:
                rows.append([0.0, 0.0, 1.0])
        return np.asarray(rows, dtype=np.float32)


def make_chunk(chunk_id: str, text: str, form: str = "10-K", section_id: str = "item_1a") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        ticker="AAPL",
        form=form,
        filing_date=date(2025, 10, 31),
        accession_number="0001",
        section_id=section_id,
        section_title="Risk Factors",
        text=text,
        source_url="https://www.sec.gov/example",
    )


def test_normalize_rows_produces_unit_vectors() -> None:
    matrix = np.asarray([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32)
    normalized = _normalize_rows(matrix)
    assert np.linalg.norm(normalized[0]) == pytest.approx(1.0)
    assert np.linalg.norm(normalized[1]) == pytest.approx(1.0)


def test_vector_search_returns_semantically_matching_chunk() -> None:
    chunks = [
        make_chunk("risk", "Supplier production disruptions could affect availability."),
        make_chunk("revenue", "Revenue and sales increased during the fiscal year.", section_id="item_7"),
    ]
    retriever = VectorRetriever(chunks, FakeEncoder())

    results = retriever.search("manufacturing partner dependency", top_k=1)

    assert results[0].chunk.chunk_id == "risk"
    assert results[0].vector_score == pytest.approx(1.0)


def test_vector_search_supports_form_and_section_filters() -> None:
    chunks = [
        make_chunk("annual", "Supplier production disruption.", form="10-K", section_id="item_1a"),
        make_chunk("quarterly", "Supplier production disruption.", form="10-Q", section_id="part_ii_item_1a"),
    ]
    retriever = VectorRetriever(chunks, FakeEncoder())

    results = retriever.search("manufacturing risk", form="10-Q", section_id="part_ii_item_1a")

    assert [result.chunk.chunk_id for result in results] == ["quarterly"]


def test_vector_search_rejects_empty_query() -> None:
    retriever = VectorRetriever([make_chunk("risk", "Supplier disruption")], FakeEncoder())
    with pytest.raises(ValueError, match="empty"):
        retriever.search("   ")
