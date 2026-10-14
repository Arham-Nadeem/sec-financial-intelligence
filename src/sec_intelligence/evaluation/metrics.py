#Retrieval and citation metrics for comparing search strategies.
from __future__ import annotations

import math
import re

from sec_intelligence.models import RetrievalResult


def recall_at_k(results: list[RetrievalResult], expected_sections: set[str], k: int) -> float:
    if not expected_sections:
        return 0.0
    found = {row.chunk.section_id for row in results[:k]}
    return len(found & expected_sections) / len(expected_sections)


def reciprocal_rank(results: list[RetrievalResult], expected_sections: set[str]) -> float:
    for rank, row in enumerate(results, start=1):
        if row.chunk.section_id in expected_sections:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(results: list[RetrievalResult], expected_sections: set[str], k: int) -> float:
    relevance = [1.0 if row.chunk.section_id in expected_sections else 0.0 for row in results[:k]]
    dcg = sum(rel / math.log2(idx + 2) for idx, rel in enumerate(relevance))
    ideal = [1.0] * min(len(expected_sections), k)
    idcg = sum(rel / math.log2(idx + 2) for idx, rel in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def keyword_hit_rate(results: list[RetrievalResult], expected_keywords: set[str], k: int) -> float:
    if not expected_keywords:
        return 0.0
    text = " ".join(row.chunk.text.lower() for row in results[:k])
    hits = sum(1 for keyword in expected_keywords if keyword.lower() in text)
    return hits / len(expected_keywords)


def citation_coverage(answer: str) -> float:
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", answer) if part.strip()]
    if not sentences:
        return 0.0
    cited = sum(1 for sentence in sentences if re.search(r"\[[^\]]+\]", sentence))
    return cited / len(sentences)
