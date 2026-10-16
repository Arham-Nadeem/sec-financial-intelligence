#Test cross-encoder reranking without downloading an external model.
from datetime import date

import numpy as np

from sec_intelligence.models import Chunk, RetrievalResult
from sec_intelligence.retrieval.reranker import CrossEncoderReranker


class FakeScorer:
    def predict(self, pairs: list[tuple[str, str]], **kwargs: object) -> np.ndarray:
        return np.asarray([1.0 if "outsourcing" in passage.lower() else 0.1 for _, passage in pairs])


def make_result(chunk_id: str, text: str, score: float) -> RetrievalResult:
    chunk = Chunk(
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
    return RetrievalResult(chunk=chunk, score=score)


def test_reranker_promotes_more_relevant_passage() -> None:
    results = [
        make_result("a", "General competitive risk discussion.", 0.9),
        make_result("b", "The company depends on outsourcing partners for manufacturing.", 0.5),
    ]

    reranked = CrossEncoderReranker(scorer=FakeScorer()).rerank(
        "dependence on outside manufacturers", results, top_k=2
    )

    assert reranked[0].chunk.chunk_id == "b"
    assert reranked[0].rerank_score == 1.0
