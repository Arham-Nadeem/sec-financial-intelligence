#Build source-grounded research answers and filing comparisons from retrieved SEC evidence.
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from difflib import SequenceMatcher

from sec_intelligence.models import RetrievalResult


@dataclass(frozen=True)
class Citation:
    """Source metadata attached to one evidence statement."""

    citation_id: int
    ticker: str
    form: str
    filing_date: date
    section_id: str
    section_title: str
    chunk_id: str
    source_url: str
    excerpt: str


@dataclass(frozen=True)
class GroundedAnswer:
    """Research response composed only from retrieved filing evidence."""

    query: str
    answer: str
    citations: list[Citation]
    evidence_count: int
    citation_coverage: float


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'-]*")
_FORM_PREFIX = re.compile(r"^[A-Za-z .&]+\|\s*20\d{2}\s+Form\s+10-[KQ]\s*\|\s*\d+\s*", re.IGNORECASE)
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
    "how", "in", "is", "it", "its", "of", "on", "or", "that", "the", "their", "this",
    "to", "was", "were", "what", "when", "which", "with",
}
_RISK_TERMS = {
    "risk", "risks", "supplier", "suppliers", "manufacturing", "manufacturer", "manufacturers",
    "outsourcing", "supply", "disruption", "disruptions", "dependency", "dependence", "depends",
    "geopolitical", "logistics", "contract",
}
_RISK_SECTIONS = {"item_1a", "part_ii_item_1a", "item_7", "part_i_item_2"}


def _terms(text: str) -> set[str]:
    """Return normalized content terms used to score evidence sentences."""
    return {
        token.lower()
        for token in _WORD.findall(text)
        if len(token) > 2 and token.lower() not in _STOPWORDS
    }


def _clean_sentence(text: str) -> str:
    """Remove filing page headers that can leak into extracted evidence."""
    return _FORM_PREFIX.sub("", " ".join(text.split())).strip()


def _best_sentence(query: str, text: str) -> str:
    """Select the sentence with the strongest query-term overlap from a chunk."""
    sentences = [_clean_sentence(sentence) for sentence in _SENTENCE_SPLIT.split(" ".join(text.split()))]
    sentences = [sentence for sentence in sentences if sentence]
    if not sentences:
        return _clean_sentence(text)

    query_terms = _terms(query)
    scored: list[tuple[float, int, str]] = []
    for index, sentence in enumerate(sentences):
        sentence_terms = _terms(sentence)
        overlap = len(query_terms & sentence_terms)
        coverage = overlap / max(len(query_terms), 1)
        #Prefer sentences with query overlap while mildly favoring concise evidence.
        score = overlap + coverage - min(len(sentence) / 3000.0, 0.25)
        scored.append((score, -index, sentence))
    return max(scored, key=lambda row: (row[0], row[1]))[2]


def _trim(text: str, limit: int = 420) -> str:
    """Trim long evidence without cutting the output at an arbitrary mid-word boundary."""
    clean = " ".join(text.split())
    if len(clean) <= limit:
        return clean
    shortened = clean[: limit + 1].rsplit(" ", 1)[0]
    return shortened.rstrip(" ,;:") + "..."


def _result_strength(result: RetrievalResult) -> float:
    """Use reranker relevance when available, otherwise fall back to retrieval score."""
    return result.rerank_score if result.rerank_score is not None else result.score


def _preferred_sections(query: str) -> set[str] | None:
    """Infer high-value filing sections for common risk-oriented research questions."""
    if _terms(query) & _RISK_TERMS:
        return _RISK_SECTIONS
    return None


def _filter_by_relevance(results: list[RetrievalResult], max_evidence: int) -> list[RetrievalResult]:
    """Drop weak tail results when reranker scores show a clear relevance gap."""
    if not results:
        return []
    ranked = sorted(results, key=_result_strength, reverse=True)
    reranked = [result for result in ranked if result.rerank_score is not None]
    if not reranked:
        return ranked[:max_evidence]

    best = _result_strength(reranked[0])
    #Cross-encoder scores are unbounded logits; retain evidence near the best result and avoid clearly irrelevant tails.
    threshold = max(0.25, best - 2.0) if best > 0 else best - 0.75
    selected = [result for result in ranked if _result_strength(result) >= threshold]
    return (selected or ranked[:1])[:max_evidence]


def _is_near_duplicate(text: str, prior: list[str], threshold: float = 0.90) -> bool:
    """Detect nearly identical evidence repeated across filings."""
    normalized = " ".join(text.lower().split())
    return any(SequenceMatcher(None, normalized, previous).ratio() >= threshold for previous in prior)


def _select_evidence(
    query: str,
    results: list[RetrievalResult],
    max_evidence: int,
    distinct_filings: bool = False,
) -> list[RetrievalResult]:
    """Select relevant, section-aware, non-duplicate evidence for grounded answers."""
    preferred = _preferred_sections(query)
    candidates = results
    if preferred:
        preferred_rows = [result for result in results if result.chunk.section_id in preferred]
        if preferred_rows:
            candidates = preferred_rows

    candidates = _filter_by_relevance(candidates, max(max_evidence * 3, max_evidence))
    selected: list[RetrievalResult] = []
    seen_chunks: set[str] = set()
    seen_filings: set[tuple[str, date]] = set()
    seen_texts: list[str] = []

    for result in candidates:
        chunk = result.chunk
        filing_key = (chunk.form, chunk.filing_date)
        if chunk.chunk_id in seen_chunks:
            continue
        if distinct_filings and filing_key in seen_filings:
            continue

        evidence = _best_sentence(query, chunk.text)
        if _is_near_duplicate(evidence, seen_texts):
            continue

        selected.append(result)
        seen_chunks.add(chunk.chunk_id)
        seen_filings.add(filing_key)
        seen_texts.append(" ".join(evidence.lower().split()))
        if len(selected) >= max_evidence:
            break

    if not selected and results:
        selected = [max(results, key=_result_strength)]
    return selected


def _assemble_answer(query: str, selected: list[RetrievalResult], comparison: bool = False) -> GroundedAnswer:
    """Create citation records and a readable extractive answer from selected evidence."""
    citations: list[Citation] = []
    statements: list[str] = []
    chronological = sorted(selected, key=lambda result: result.chunk.filing_date) if comparison else selected

    for citation_id, result in enumerate(chronological, start=1):
        chunk = result.chunk
        evidence = _trim(_best_sentence(query, chunk.text))
        citations.append(
            Citation(
                citation_id=citation_id,
                ticker=chunk.ticker,
                form=chunk.form,
                filing_date=chunk.filing_date,
                section_id=chunk.section_id,
                section_title=chunk.section_title,
                chunk_id=chunk.chunk_id,
                source_url=chunk.source_url,
                excerpt=evidence,
            )
        )
        if comparison:
            statements.append(f"{chunk.filing_date}: {evidence} [{citation_id}]")
        else:
            statements.append(f"{evidence} [{citation_id}]")

    answer = " ".join(statements) if statements else "No supporting SEC filing evidence was retrieved for this question."
    cited_statements = sum(1 for statement in statements if re.search(r"\[\d+\]$", statement))
    coverage = cited_statements / len(statements) if statements else 0.0
    return GroundedAnswer(
        query=query,
        answer=answer,
        citations=citations,
        evidence_count=len(citations),
        citation_coverage=coverage,
    )


def build_grounded_answer(
    query: str,
    results: list[RetrievalResult],
    max_evidence: int = 4,
) -> GroundedAnswer:
    """Create a relevance-filtered extractive answer where every claim maps to SEC evidence."""
    if not query.strip():
        raise ValueError("Query cannot be empty.")
    if max_evidence <= 0:
        raise ValueError("max_evidence must be positive.")
    if not results:
        return GroundedAnswer(
            query=query,
            answer="No supporting SEC filing evidence was retrieved for this question.",
            citations=[],
            evidence_count=0,
            citation_coverage=0.0,
        )

    selected = _select_evidence(query, results, max_evidence=max_evidence)
    return _assemble_answer(query, selected)


def compare_filings(
    query: str,
    results: list[RetrievalResult],
    max_filings: int = 4,
) -> GroundedAnswer:
    """Compare relevant, non-duplicate evidence from distinct filing dates."""
    if not query.strip():
        raise ValueError("Query cannot be empty.")
    if max_filings <= 0:
        raise ValueError("max_filings must be positive.")
    if not results:
        return build_grounded_answer(query, results, max_evidence=max_filings)

    selected = _select_evidence(query, results, max_evidence=max_filings, distinct_filings=True)
    return _assemble_answer(query, selected, comparison=True)


def validate_citations(answer: GroundedAnswer) -> dict[str, float | int | bool]:
    """Check that answer citation markers map one-to-one to available source records."""
    markers = [int(value) for value in re.findall(r"\[(\d+)\]", answer.answer)]
    valid_ids = {citation.citation_id for citation in answer.citations}
    valid_markers = [marker for marker in markers if marker in valid_ids]
    precision = len(valid_markers) / len(markers) if markers else 0.0
    expected = set(range(1, len(answer.citations) + 1))
    coverage = len(set(valid_markers)) / len(expected) if expected else 0.0
    return {
        "citation_markers": len(markers),
        "valid_citations": len(valid_markers),
        "citation_precision": precision,
        "citation_coverage": coverage,
        "all_citations_valid": bool(markers) and precision == 1.0 and coverage == 1.0,
    }
