#Local BM25 plus semantic-vector retrieval with optional cross-encoder reranking.
from __future__ import annotations

import math
import re
from dataclasses import dataclass

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from sec_intelligence.models import Chunk, RetrievalResult


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def minmax(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    low = float(values.min())
    high = float(values.max())
    if math.isclose(low, high):
        return np.ones_like(values) if high > 0 else np.zeros_like(values)
    return (values - low) / (high - low)


@dataclass
class HybridRetriever:
    chunks: list[Chunk]
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    vector_weight: float = 0.55
    bm25_weight: float = 0.45
    use_reranker: bool = True

    def __post_init__(self) -> None:
        if not self.chunks:
            raise ValueError("At least one chunk is required to build a retriever.")
        self.encoder = SentenceTransformer(self.embedding_model_name)
        corpus = [tokenize(chunk.text) for chunk in self.chunks]
        self.bm25 = BM25Okapi(corpus)
        embeddings = self.encoder.encode(
            [chunk.text for chunk in self.chunks],
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        self.reranker = CrossEncoder(self.reranker_model_name) if self.use_reranker else None

    def search(
        self,
        query: str,
        candidate_k: int = 20,
        top_k: int = 6,
        mode: str = "hybrid",
        apply_reranker: bool | None = None,
    ) -> list[RetrievalResult]:
        candidate_k = min(candidate_k, len(self.chunks))
        query_vector = self.encoder.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        ).astype("float32")
        vector_scores, vector_ids = self.index.search(query_vector, candidate_k)
        vector_lookup = {
            int(idx): float(score) for idx, score in zip(vector_ids[0], vector_scores[0]) if idx >= 0
        }

        bm25_scores = np.asarray(self.bm25.get_scores(tokenize(query)), dtype=float)
        bm25_order = np.argsort(bm25_scores)[::-1][:candidate_k]
        candidate_ids = sorted(set(vector_lookup) | {int(idx) for idx in bm25_order})

        raw_vector = np.asarray([vector_lookup.get(idx, 0.0) for idx in candidate_ids], dtype=float)
        raw_bm25 = np.asarray([bm25_scores[idx] for idx in candidate_ids], dtype=float)
        norm_vector = minmax(raw_vector)
        norm_bm25 = minmax(raw_bm25)
        if mode == "vector":
            hybrid = norm_vector
        elif mode == "bm25":
            hybrid = norm_bm25
        elif mode == "hybrid":
            hybrid = self.vector_weight * norm_vector + self.bm25_weight * norm_bm25
        else:
            raise ValueError("mode must be one of: vector, bm25, hybrid")

        rows = []
        for pos, idx in enumerate(candidate_ids):
            rows.append(
                RetrievalResult(
                    chunk=self.chunks[idx],
                    score=float(hybrid[pos]),
                    vector_score=float(raw_vector[pos]),
                    bm25_score=float(raw_bm25[pos]),
                )
            )
        rows.sort(key=lambda row: row.score, reverse=True)
        rows = rows[:candidate_k]

        should_rerank = self.use_reranker if apply_reranker is None else apply_reranker
        if self.reranker and should_rerank and rows:
            pairs = [(query, row.chunk.text) for row in rows]
            rerank_scores = self.reranker.predict(pairs)
            for row, score in zip(rows, rerank_scores):
                row.rerank_score = float(score)
            rows.sort(key=lambda row: row.rerank_score or float("-inf"), reverse=True)

        return rows[:top_k]
