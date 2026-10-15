#Provide a dependency-light BM25 baseline over parsed SEC filing chunks.
from __future__ import annotations

from collections import Counter
import json
import math
import re
from pathlib import Path

from sec_intelligence.models import Chunk, RetrievalResult


def tokenize(text: str) -> list[str]:
    """Normalize filing text into simple alphanumeric search tokens."""
    return re.findall(r"[a-z0-9]+", text.lower())


def load_chunks(path: Path) -> list[Chunk]:
    """Load newline-delimited chunk records produced by the filing parser."""
    if not path.exists():
        raise FileNotFoundError(f"Chunk file not found: {path}")

    chunks: list[Chunk] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                chunks.append(Chunk.model_validate(json.loads(stripped)))
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"Invalid chunk record on line {line_number}: {path}") from exc
    return chunks


class BM25Retriever:
    """Rank SEC filing chunks with the Okapi BM25 relevance formula."""

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        if not chunks:
            raise ValueError("At least one chunk is required to build a BM25 index.")
        if k1 <= 0:
            raise ValueError("k1 must be positive.")
        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1.")

        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self._documents = [tokenize(chunk.text) for chunk in chunks]
        self._term_frequencies = [Counter(document) for document in self._documents]
        self._document_lengths = [len(document) for document in self._documents]
        self._average_document_length = sum(self._document_lengths) / len(self._document_lengths)
        self._document_frequency = self._build_document_frequency()

    def _build_document_frequency(self) -> Counter[str]:
        frequencies: Counter[str] = Counter()
        for document in self._documents:
            frequencies.update(set(document))
        return frequencies

    def _idf(self, term: str) -> float:
        document_count = len(self._documents)
        frequency = self._document_frequency.get(term, 0)
        return math.log(1 + (document_count - frequency + 0.5) / (frequency + 0.5))

    def _score_document(self, query_tokens: list[str], document_index: int) -> float:
        frequencies = self._term_frequencies[document_index]
        document_length = self._document_lengths[document_index]
        length_ratio = document_length / self._average_document_length if self._average_document_length else 0
        score = 0.0

        for term in set(query_tokens):
            term_frequency = frequencies.get(term, 0)
            if term_frequency == 0:
                continue
            numerator = term_frequency * (self.k1 + 1)
            denominator = term_frequency + self.k1 * (1 - self.b + self.b * length_ratio)
            score += self._idf(term) * (numerator / denominator)
        return score

    def search(
        self,
        query: str,
        top_k: int = 5,
        form: str | None = None,
        section_id: str | None = None,
    ) -> list[RetrievalResult]:
        """Return the highest-scoring chunks with optional filing/section filters."""
        query_tokens = tokenize(query)
        if not query_tokens:
            raise ValueError("Query must contain at least one searchable token.")
        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        normalized_form = form.upper() if form else None
        normalized_section = section_id.lower() if section_id else None
        candidates: list[tuple[int, float]] = []

        for index, chunk in enumerate(self.chunks):
            if normalized_form and chunk.form.upper() != normalized_form:
                continue
            if normalized_section and chunk.section_id.lower() != normalized_section:
                continue
            candidates.append((index, self._score_document(query_tokens, index)))

        candidates.sort(key=lambda row: row[1], reverse=True)
        return [
            RetrievalResult(
                chunk=self.chunks[index],
                score=score,
                bm25_score=score,
            )
            for index, score in candidates[:top_k]
        ]
