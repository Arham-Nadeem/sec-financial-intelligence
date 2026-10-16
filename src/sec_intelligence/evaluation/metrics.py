#Compute ranking-quality and evidence-coverage metrics for SEC retrieval benchmarks.
from __future__ import annotations

import math

from sec_intelligence.models import RetrievalResult


def _normalized_sections(expected_sections: set[str]) -> set[str]:
    return {section.lower() for section in expected_sections}


def _relevant(result: RetrievalResult, expected_sections: set[str]) -> bool:
    return result.chunk.section_id.lower() in _normalized_sections(expected_sections)


def recall_at_k(results: list[RetrievalResult], expected_sections: set[str], k: int) -> float:
    """Measure how many unique expected filing sections appear in the top-k results."""
    if not expected_sections:
        return 0.0
    found = {row.chunk.section_id.lower() for row in results[:k]}
    expected = _normalized_sections(expected_sections)
    return len(found & expected) / len(expected)


def reciprocal_rank(results: list[RetrievalResult], expected_sections: set[str]) -> float:
    """Return the reciprocal rank of the first section-relevant result."""
    if not expected_sections:
        return 0.0
    for rank, row in enumerate(results, start=1):
        if _relevant(row, expected_sections):
            return 1.0 / rank
    return 0.0


def ndcg_at_k(results: list[RetrievalResult], expected_sections: set[str], k: int) -> float:
    """Measure unique-section ranking quality with binary relevance in the top-k results."""
    if not expected_sections:
        return 0.0

    expected = _normalized_sections(expected_sections)
    seen_relevant: set[str] = set()
    gains: list[float] = []

    #Give credit only for the first occurrence of each expected section so repeated
    #chunks from one filing section cannot inflate nDCG above 1.0.
    for row in results[:k]:
        section = row.chunk.section_id.lower()
        if section in expected and section not in seen_relevant:
            gains.append(1.0)
            seen_relevant.add(section)
        else:
            gains.append(0.0)

    dcg = sum(gain / math.log2(rank + 2) for rank, gain in enumerate(gains))
    ideal_count = min(len(expected), k)
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_count))
    return dcg / idcg if idcg else 0.0


def keyword_hit_rate(
    results: list[RetrievalResult], expected_keywords: set[str], k: int
) -> float:
    """Measure whether expected evidence terms occur anywhere in the top-k context."""
    if not expected_keywords:
        return 0.0
    combined = " ".join(row.chunk.text.lower() for row in results[:k])
    hits = sum(1 for keyword in expected_keywords if keyword.lower() in combined)
    return hits / len(expected_keywords)


def relevant_result_rate(
    results: list[RetrievalResult], expected_sections: set[str], k: int
) -> float:
    """Measure the share of top-k chunks drawn from labeled relevant sections."""
    window = results[:k]
    if not window or not expected_sections:
        return 0.0
    relevant = sum(1 for result in window if _relevant(result, expected_sections))
    return relevant / len(window)
