#Rerank retrieved SEC filing evidence with a local cross-encoder model.
from __future__ import annotations

from typing import Protocol

import numpy as np

from sec_intelligence.models import RetrievalResult


DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class PairScorer(Protocol):
    """Minimal pair-scoring interface used by the reranker and unit tests."""

    def predict(self, pairs: list[tuple[str, str]], **kwargs: object) -> np.ndarray:
        ...


class CrossEncoderScorer:
    """Lazy wrapper around a local sentence-transformers cross encoder."""

    def __init__(self, model_name: str = DEFAULT_RERANKER) -> None:
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def predict(self, pairs: list[tuple[str, str]], **kwargs: object) -> np.ndarray:
        return np.asarray(
            self.model.predict(
                pairs,
                show_progress_bar=bool(kwargs.get("show_progress_bar", False)),
            ),
            dtype=float,
        )


class CrossEncoderReranker:
    """Score query-passage pairs jointly and reorder first-stage retrieval candidates."""

    def __init__(self, scorer: PairScorer | None = None, model_name: str = DEFAULT_RERANKER) -> None:
        self.scorer = scorer or CrossEncoderScorer(model_name)

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """Return candidates ordered by cross-encoder relevance score."""
        if not query.strip():
            raise ValueError("Query cannot be empty.")
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        if not results:
            return []

        pairs = [(query, result.chunk.text) for result in results]
        scores = np.asarray(self.scorer.predict(pairs, show_progress_bar=False), dtype=float).reshape(-1)
        if len(scores) != len(results):
            raise ValueError("Reranker score count must match candidate count.")

        reranked = list(results)
        for result, score in zip(reranked, scores):
            result.rerank_score = float(score)
        reranked.sort(
            key=lambda result: result.rerank_score if result.rerank_score is not None else float("-inf"),
            reverse=True,
        )
        return reranked[:top_k]
