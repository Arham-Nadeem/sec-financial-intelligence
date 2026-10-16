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

_CONCEPT_EXPANSIONS = {
    "supplier": {"supplier", "suppliers", "supply", "outsourcing", "partner", "partners", "contract", "component", "components"},
    "suppliers": {"supplier", "suppliers", "supply", "outsourcing", "partner", "partners", "contract", "component", "components"},
    "manufacturing": {"manufacturing", "manufacture", "manufacturer", "manufacturers", "assembly", "logistics", "production"},
    "manufacturer": {"manufacturing", "manufacture", "manufacturer", "manufacturers", "assembly", "logistics", "production"},
    "risk": {"risk", "risks", "disruption", "disruptions", "adverse", "interruption", "interruptions", "dependency", "dependence", "depends"},
    "risks": {"risk", "risks", "disruption", "disruptions", "adverse", "interruption", "interruptions", "dependency", "dependence", "depends"},
}


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


def _expanded_query_terms(query: str) -> set[str]:
    """Expand common finance-research concepts into language used across SEC filings."""
    base = _terms(query)
    expanded = set(base)
    for term in base:
        expanded.update(_CONCEPT_EXPANSIONS.get(term, set()))
    return expanded




def _best_comparison_sentence(query: str, text: str) -> str:
    """Select substantive comparison evidence using expanded filing-language concepts."""
    sentences = [_clean_sentence(sentence) for sentence in _SENTENCE_SPLIT.split(" ".join(text.split()))]
    sentences = [sentence for sentence in sentences if sentence]
    if not sentences:
        return _clean_sentence(text)

    query_terms = _expanded_query_terms(query)
    scored: list[tuple[float, int, str]] = []
    for index, sentence in enumerate(sentences):
        sentence_terms = _terms(sentence)
        overlap = len(query_terms & sentence_terms)
        coverage = overlap / max(len(query_terms), 1)
        word_count = len(_WORD.findall(sentence))
        #Reject page headers/item labels and reward sentences carrying comparison concepts.
        substantive_bonus = 0.35 if word_count >= 10 else -1.0
        score = overlap + coverage + substantive_bonus - min(len(sentence) / 4000.0, 0.20)
        scored.append((score, -index, sentence))
    return max(scored, key=lambda row: (row[0], row[1]))[2]




def _required_concept_groups(query: str) -> list[set[str]]:
    """Return core topic groups that comparison evidence must substantively match."""
    query_terms = _terms(query)
    groups: list[set[str]] = []
    if query_terms & {"supplier", "suppliers", "supply"}:
        groups.append({"supplier", "suppliers", "supply", "outsourcing", "partner", "partners", "contract", "component", "components"})
    if query_terms & {"manufacturing", "manufacturer", "manufacturers", "manufacture"}:
        groups.append({"manufacturing", "manufacture", "manufacturer", "manufacturers", "assembly", "logistics", "production"})
    return groups

def _is_substantive_comparison_evidence(query: str, text: str) -> bool:
    """Reject fragments and evidence that misses the query's core comparison topics."""
    clean = _clean_sentence(text)
    words = _WORD.findall(clean)
    if len(words) < 10:
        return False
    evidence_terms = _terms(clean)
    if not (_expanded_query_terms(query) & evidence_terms):
        return False

    #For explicit supplier/manufacturing comparisons, require every named core topic to be present.
    #This prevents generic regulatory or operational-risk passages from being forced into the timeline.
    for group in _required_concept_groups(query):
        if not (group & evidence_terms):
            return False
    return True

def _comparison_focus_score(query: str, result: RetrievalResult) -> tuple[float, float]:
    """Score evidence for cross-filing comparison using concept coverage plus model relevance."""
    evidence = _best_comparison_sentence(query, result.chunk.text)
    evidence_terms = _terms(evidence)
    expanded = _expanded_query_terms(query)
    overlap = len(expanded & evidence_terms)
    concept_coverage = overlap / max(len(expanded), 1)

    #Reward explicit dependency/supplier language because it is more useful than generic manufacturing mentions.
    dependency_terms = {"supplier", "suppliers", "outsourcing", "partner", "partners", "contract", "depends", "dependence"}
    dependency_bonus = 0.35 * len(dependency_terms & evidence_terms)
    model_score = _result_strength(result)
    return (overlap + concept_coverage + dependency_bonus, model_score)


def _select_comparison_evidence(
    query: str,
    results: list[RetrievalResult],
    max_filings: int,
) -> list[RetrievalResult]:
    """Choose the strongest concept-matched evidence from distinct filing dates."""
    preferred = _preferred_sections(query)
    candidates = results
    if preferred:
        preferred_rows = [result for result in results if result.chunk.section_id in preferred]
        if preferred_rows:
            candidates = preferred_rows

    by_filing: dict[tuple[str, date], list[RetrievalResult]] = {}
    for result in candidates:
        key = (result.chunk.form, result.chunk.filing_date)
        by_filing.setdefault(key, []).append(result)

    representatives: list[RetrievalResult] = []
    for filing_results in by_filing.values():
        #Choose from topic-valid passages inside each filing instead of choosing globally first.
        #This keeps an older filing in the comparison when its best overall chunk is too generic.
        substantive = [
            result
            for result in filing_results
            if _is_substantive_comparison_evidence(
                query,
                _best_comparison_sentence(query, result.chunk.text),
            )
        ]
        if substantive:
            representatives.append(
                max(substantive, key=lambda result: _comparison_focus_score(query, result))
            )

    if not representatives:
        return []

    representatives.sort(
        key=lambda result: (
            _comparison_focus_score(query, result),
            result.chunk.filing_date,
        ),
        reverse=True,
    )

    #One representative already exists per filing date. Keep similar wording across years because
    #that similarity is itself useful evidence when comparing how disclosures changed over time.
    return representatives[:max_filings]


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
        evidence_source = _best_comparison_sentence(query, chunk.text) if comparison else _best_sentence(query, chunk.text)
        evidence = _trim(evidence_source)
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

    selected = _select_comparison_evidence(query, results, max_filings=max_filings)
    if not selected:
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
