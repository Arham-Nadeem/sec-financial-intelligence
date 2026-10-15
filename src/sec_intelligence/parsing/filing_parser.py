#Parse SEC filing HTML into clean text and filing-aware sections.
from __future__ import annotations

import re
from dataclasses import dataclass

from bs4 import BeautifulSoup

from sec_intelligence.models import FilingSection


@dataclass(frozen=True)
class HeadingCandidate:
    section_id: str
    title: str
    start: int
    end: int


#Canonical 10-K items used to identify substantive filing sections.
TEN_K_ITEMS: tuple[tuple[str, str, str], ...] = (
    ("item_1", "Item 1 - Business", r"item\s+1\.?\s*(?:[-–—:]\s*)?(?:business)?"),
    ("item_1a", "Item 1A - Risk Factors", r"item\s+1a\.?\s*(?:[-–—:]\s*)?(?:risk\s+factors)?"),
    ("item_1b", "Item 1B - Unresolved Staff Comments", r"item\s+1b\.?\s*(?:[-–—:]\s*)?(?:unresolved\s+staff\s+comments)?"),
    ("item_1c", "Item 1C - Cybersecurity", r"item\s+1c\.?\s*(?:[-–—:]\s*)?(?:cybersecurity)?"),
    ("item_2", "Item 2 - Properties", r"item\s+2\.?\s*(?:[-–—:]\s*)?(?:properties)?"),
    ("item_3", "Item 3 - Legal Proceedings", r"item\s+3\.?\s*(?:[-–—:]\s*)?(?:legal\s+proceedings)?"),
    ("item_4", "Item 4 - Mine Safety Disclosures", r"item\s+4\.?\s*(?:[-–—:]\s*)?(?:mine\s+safety\s+disclosures)?"),
    ("item_5", "Item 5 - Market for Registrant's Common Equity", r"item\s+5\.?\s*(?:[-–—:]\s*)?(?:market\b.*)?"),
    ("item_6", "Item 6 - Reserved", r"item\s+6\.?\s*(?:[-–—:]\s*)?(?:reserved)?"),
    ("item_7", "Item 7 - Management's Discussion and Analysis", r"item\s+7\.?\s*(?:[-–—:]\s*)?(?:management['’]s\s+discussion.*)?"),
    ("item_7a", "Item 7A - Quantitative and Qualitative Disclosures", r"item\s+7a\.?\s*(?:[-–—:]\s*)?(?:quantitative\s+and\s+qualitative.*)?"),
    ("item_8", "Item 8 - Financial Statements and Supplementary Data", r"item\s+8\.?\s*(?:[-–—:]\s*)?(?:financial\s+statements.*)?"),
    ("item_9", "Item 9 - Changes in and Disagreements With Accountants", r"item\s+9\.?\s*(?:[-–—:]\s*)?(?:changes\s+in\s+and\s+disagreements.*)?"),
    ("item_9a", "Item 9A - Controls and Procedures", r"item\s+9a\.?\s*(?:[-–—:]\s*)?(?:controls\s+and\s+procedures)?"),
    ("item_9b", "Item 9B - Other Information", r"item\s+9b\.?\s*(?:[-–—:]\s*)?(?:other\s+information)?"),
    ("item_9c", "Item 9C - Foreign Jurisdictions", r"item\s+9c\.?\s*(?:[-–—:]\s*)?(?:disclosure\s+regarding.*)?"),
    ("item_10", "Item 10 - Directors, Executive Officers and Corporate Governance", r"item\s+10\.?\s*(?:[-–—:]\s*)?(?:directors.*)?"),
    ("item_11", "Item 11 - Executive Compensation", r"item\s+11\.?\s*(?:[-–—:]\s*)?(?:executive\s+compensation)?"),
    ("item_12", "Item 12 - Security Ownership", r"item\s+12\.?\s*(?:[-–—:]\s*)?(?:security\s+ownership.*)?"),
    ("item_13", "Item 13 - Certain Relationships and Related Transactions", r"item\s+13\.?\s*(?:[-–—:]\s*)?(?:certain\s+relationships.*)?"),
    ("item_14", "Item 14 - Principal Accountant Fees and Services", r"item\s+14\.?\s*(?:[-–—:]\s*)?(?:principal\s+accountant.*)?"),
    ("item_15", "Item 15 - Exhibits and Financial Statement Schedules", r"item\s+15\.?\s*(?:[-–—:]\s*)?(?:exhibits.*)?"),
    ("item_16", "Item 16 - Form 10-K Summary", r"item\s+16\.?\s*(?:[-–—:]\s*)?(?:form\s+10-k\s+summary)?"),
)

#10-Q item numbering resets between Part I and Part II, so part context is tracked separately.
TEN_Q_PART_I: dict[str, tuple[str, str]] = {
    "1": ("part_i_item_1", "Part I Item 1 - Financial Statements"),
    "2": ("part_i_item_2", "Part I Item 2 - Management's Discussion and Analysis"),
    "3": ("part_i_item_3", "Part I Item 3 - Quantitative and Qualitative Disclosures"),
    "4": ("part_i_item_4", "Part I Item 4 - Controls and Procedures"),
}
TEN_Q_PART_II: dict[str, tuple[str, str]] = {
    "1": ("part_ii_item_1", "Part II Item 1 - Legal Proceedings"),
    "1a": ("part_ii_item_1a", "Part II Item 1A - Risk Factors"),
    "2": ("part_ii_item_2", "Part II Item 2 - Unregistered Sales and Use of Proceeds"),
    "3": ("part_ii_item_3", "Part II Item 3 - Defaults Upon Senior Securities"),
    "4": ("part_ii_item_4", "Part II Item 4 - Mine Safety Disclosures"),
    "5": ("part_ii_item_5", "Part II Item 5 - Other Information"),
    "6": ("part_ii_item_6", "Part II Item 6 - Exhibits"),
}


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text("\n")
    text = text.replace("\xa0", " ")
    text = re.sub(r"[\t\r ]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_into_sections(text: str, form: str) -> list[FilingSection]:
    normalized_form = form.upper()
    if normalized_form == "10-K":
        candidates = _find_10k_candidates(text)
    elif normalized_form == "10-Q":
        candidates = _find_10q_candidates(text)
    else:
        candidates = []

    if not candidates:
        return [FilingSection(section_id="full_filing", title="Full Filing", text=text)]

    selected = _select_substantive_occurrences(text, candidates)
    sections = _build_sections(text, selected)
    return sections or [FilingSection(section_id="full_filing", title="Full Filing", text=text)]


def _find_10k_candidates(text: str) -> list[HeadingCandidate]:
    candidates: list[HeadingCandidate] = []
    for start, end, line in _iter_lines(text):
        cleaned = _clean_heading(line)
        if not _looks_like_heading(cleaned):
            continue
        for section_id, title, pattern in TEN_K_ITEMS:
            if re.fullmatch(pattern, cleaned, flags=re.IGNORECASE):
                candidates.append(HeadingCandidate(section_id, title, start, end))
                break
    return candidates


def _find_10q_candidates(text: str) -> list[HeadingCandidate]:
    candidates: list[HeadingCandidate] = []
    current_part: str | None = None

    for start, end, line in _iter_lines(text):
        cleaned = _clean_heading(line)
        if re.fullmatch(r"part\s+i\.?", cleaned, flags=re.IGNORECASE):
            current_part = "I"
            continue
        if re.fullmatch(r"part\s+ii\.?", cleaned, flags=re.IGNORECASE):
            current_part = "II"
            continue
        if not _looks_like_heading(cleaned):
            continue

        match = re.fullmatch(r"item\s+(1a|[1-6])\.?\s*(?:[-–—:]\s*)?.*", cleaned, flags=re.IGNORECASE)
        if not match or current_part is None:
            continue

        item = match.group(1).lower()
        mapping = TEN_Q_PART_I if current_part == "I" else TEN_Q_PART_II
        if item in mapping:
            section_id, title = mapping[item]
            candidates.append(HeadingCandidate(section_id, title, start, end))

    return candidates


def _select_substantive_occurrences(
    text: str, candidates: list[HeadingCandidate]
) -> list[HeadingCandidate]:
    candidates = sorted(candidates, key=lambda row: row.start)
    grouped: dict[str, list[HeadingCandidate]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.section_id, []).append(candidate)

    selected: list[HeadingCandidate] = []
    for occurrences in grouped.values():
        #TOC entries usually have another recognized heading immediately after them.
        best = max(
            occurrences,
            key=lambda candidate: _candidate_body_length(text, candidate, candidates),
        )
        selected.append(best)
    return sorted(selected, key=lambda row: row.start)


def _candidate_body_length(
    text: str, candidate: HeadingCandidate, all_candidates: list[HeadingCandidate]
) -> int:
    next_starts = [row.start for row in all_candidates if row.start > candidate.start]
    end = min(next_starts) if next_starts else len(text)
    return max(0, end - candidate.end)


def _build_sections(text: str, candidates: list[HeadingCandidate]) -> list[FilingSection]:
    sections: list[FilingSection] = []
    for index, candidate in enumerate(candidates):
        end = candidates[index + 1].start if index + 1 < len(candidates) else len(text)
        section_text = text[candidate.start:end].strip()
        word_count = len(section_text.split())
        if word_count < 40:
            continue
        sections.append(
            FilingSection(
                section_id=candidate.section_id,
                title=candidate.title,
                text=section_text,
            )
        )
    return sections


def _iter_lines(text: str):
    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped:
            leading = len(line) - len(line.lstrip())
            start = offset + leading
            end = start + len(stripped)
            yield start, end, stripped
        offset += len(line)


def _clean_heading(line: str) -> str:
    line = re.sub(r"\s+", " ", line).strip()
    line = line.strip("·•|_")
    return line.strip()


def _looks_like_heading(line: str) -> bool:
    if len(line) > 180:
        return False
    return bool(re.match(r"^(?:part\s+i{1,2}\b|item\s+\d)", line, flags=re.IGNORECASE))
