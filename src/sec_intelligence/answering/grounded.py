#Compose citation-first research responses from retrieved filing evidence.
from __future__ import annotations

import re

from sec_intelligence.models import RetrievalResult


def _best_sentence(text: str, query: str) -> str:
    query_terms = set(re.findall(r"[a-z0-9]+", query.lower()))
    sentences = re.split(r"(?<=[.!?])\s+", text)
    scored = []
    for sentence in sentences:
        terms = set(re.findall(r"[a-z0-9]+", sentence.lower()))
        scored.append((len(query_terms & terms), sentence.strip()))
    scored.sort(key=lambda row: row[0], reverse=True)
    return scored[0][1] if scored else text[:500]


def compose_extractive_answer(query: str, results: list[RetrievalResult], max_evidence: int = 4) -> str:
    if not results:
        return "No sufficiently relevant filing evidence was retrieved."

    lines = ["Evidence-grounded findings:"]
    for idx, result in enumerate(results[:max_evidence], start=1):
        sentence = _best_sentence(result.chunk.text, query)
        citation = (
            f"[{result.chunk.form} {result.chunk.filing_date} | "
            f"{result.chunk.section_title} | {result.chunk.chunk_id}]"
        )
        lines.append(f"{idx}. {sentence} {citation}")
    return "\n".join(lines)
