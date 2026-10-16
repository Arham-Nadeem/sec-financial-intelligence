#Test weighted reciprocal-rank fusion for SEC filing retrieval.
from datetime import date

import numpy as np

from sec_intelligence.models import Chunk
from sec_intelligence.retrieval.bm25 import BM25Retriever
from sec_intelligence.retrieval.hybrid import HybridRetriever
from sec_intelligence.retrieval.vector import VectorRetriever


class FakeEncoder:
    def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
        rows = []
        for text in texts:
            lowered = text.lower()
            rows.append([
                float("supplier" in lowered or "outside" in lowered),
                float("revenue" in lowered),
                1.0,
            ])
        return np.asarray(rows, dtype=np.float32)


def make_chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        ticker="AAPL",
        form="10-K",
        filing_date=date(2025, 10, 31),
        accession_number="0001",
        section_id="item_1a",
        section_title="Item 1A - Risk Factors",
        text=text,
        source_url="https://www.sec.gov/example",
    )


def test_hybrid_fuses_bm25_and_vector_results() -> None:
    chunks = [
        make_chunk("a", "Supplier manufacturing risk can disrupt production."),
        make_chunk("b", "Revenue increased during the fiscal year."),
        make_chunk("c", "Outside partners assemble company products."),
    ]
    bm25 = BM25Retriever(chunks)
    vector = VectorRetriever(chunks, FakeEncoder())
    hybrid = HybridRetriever(bm25, vector)

    results = hybrid.search("outside supplier dependence", top_k=3, candidate_k=3)

    assert len(results) == 3
    assert {result.chunk.chunk_id for result in results[:2]} == {"a", "c"}
    assert results[0].score >= results[1].score


def test_hybrid_rejects_invalid_weights() -> None:
    chunks = [make_chunk("a", "Supplier manufacturing risk.")]
    bm25 = BM25Retriever(chunks)
    vector = VectorRetriever(chunks, FakeEncoder())

    try:
        HybridRetriever(bm25, vector, bm25_weight=0, vector_weight=0)
    except ValueError as exc:
        assert "weight" in str(exc).lower()
    else:
        raise AssertionError("Expected invalid zero weights to raise ValueError.")
