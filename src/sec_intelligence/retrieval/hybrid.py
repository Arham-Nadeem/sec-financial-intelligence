#Combine lexical and semantic SEC retrieval with weighted reciprocal-rank fusion.
from __future__ import annotations

from sec_intelligence.models import RetrievalResult
from sec_intelligence.retrieval.bm25 import BM25Retriever
from sec_intelligence.retrieval.vector import VectorRetriever


class HybridRetriever:
    """Fuse BM25 and vector rankings without assuming comparable raw score scales."""

    def __init__(
        self,
        bm25: BM25Retriever,
        vector: VectorRetriever,
        bm25_weight: float = 0.45,
        vector_weight: float = 0.55,
        rrf_k: int = 60,
    ) -> None:
        if bm25_weight < 0 or vector_weight < 0:
            raise ValueError("Retrieval weights cannot be negative.")
        if bm25_weight + vector_weight == 0:
            raise ValueError("At least one retrieval weight must be positive.")
        if rrf_k <= 0:
            raise ValueError("rrf_k must be positive.")
        if [chunk.chunk_id for chunk in bm25.chunks] != [chunk.chunk_id for chunk in vector.chunks]:
            raise ValueError("BM25 and vector retrievers must use the same chunk collection.")

        total = bm25_weight + vector_weight
        self.bm25 = bm25
        self.vector = vector
        self.bm25_weight = bm25_weight / total
        self.vector_weight = vector_weight / total
        self.rrf_k = rrf_k
        self.chunks = bm25.chunks

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 30,
        form: str | None = None,
        section_id: str | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve candidates from both engines and fuse them with weighted RRF."""
        if not query.strip():
            raise ValueError("Query cannot be empty.")
        if top_k <= 0 or candidate_k <= 0:
            raise ValueError("top_k and candidate_k must be positive.")
        candidate_k = max(candidate_k, top_k)

        bm25_results = self.bm25.search(query, candidate_k, form=form, section_id=section_id)
        vector_results = self.vector.search(query, candidate_k, form=form, section_id=section_id)

        rows: dict[str, RetrievalResult] = {}
        fusion_scores: dict[str, float] = {}

        for rank, result in enumerate(bm25_results, start=1):
            chunk_id = result.chunk.chunk_id
            rows[chunk_id] = RetrievalResult(
                chunk=result.chunk,
                score=0.0,
                bm25_score=result.bm25_score,
            )
            fusion_scores[chunk_id] = fusion_scores.get(chunk_id, 0.0) + (
                self.bm25_weight / (self.rrf_k + rank)
            )

        for rank, result in enumerate(vector_results, start=1):
            chunk_id = result.chunk.chunk_id
            if chunk_id not in rows:
                rows[chunk_id] = RetrievalResult(
                    chunk=result.chunk,
                    score=0.0,
                    vector_score=result.vector_score,
                )
            else:
                rows[chunk_id].vector_score = result.vector_score
            fusion_scores[chunk_id] = fusion_scores.get(chunk_id, 0.0) + (
                self.vector_weight / (self.rrf_k + rank)
            )

        for chunk_id, result in rows.items():
            result.score = fusion_scores[chunk_id]

        ranked = sorted(rows.values(), key=lambda result: result.score, reverse=True)
        return ranked[:top_k]
