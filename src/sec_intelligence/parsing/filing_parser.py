#Parse SEC filing HTML into clean text and filing-aware sections.
from __future__ import annotations

import re

from bs4 import BeautifulSoup

from sec_intelligence.models import FilingSection


SECTION_PATTERNS = {
    "10-K": [
        ("item_1", r"item\s+1\.?\s+business"),
        ("item_1a", r"item\s+1a\.?\s+risk\s+factors"),
        ("item_1b", r"item\s+1b\.?\s+unresolved\s+staff\s+comments"),
        ("item_2", r"item\s+2\.?\s+properties"),
        ("item_3", r"item\s+3\.?\s+legal\s+proceedings"),
        ("item_5", r"item\s+5\.?\s+market"),
        ("item_7", r"item\s+7\.?\s+management[’']?s\s+discussion"),
        ("item_7a", r"item\s+7a\.?\s+quantitative\s+and\s+qualitative"),
        ("item_8", r"item\s+8\.?\s+financial\s+statements"),
    ],
    "10-Q": [
        ("part_i_item_1", r"part\s+i.*?item\s+1\.?\s+financial\s+statements"),
        ("part_i_item_2", r"item\s+2\.?\s+management[’']?s\s+discussion"),
        ("part_i_item_3", r"item\s+3\.?\s+quantitative\s+and\s+qualitative"),
        ("part_i_item_4", r"item\s+4\.?\s+controls\s+and\s+procedures"),
        ("part_ii_item_1a", r"item\s+1a\.?\s+risk\s+factors"),
    ],
}


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text("\n")
    text = re.sub(r"[\t\r ]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_into_sections(text: str, form: str) -> list[FilingSection]:
    patterns = SECTION_PATTERNS.get(form, [])
    lowered = text.lower()
    candidates: list[tuple[int, str, str]] = []

    for section_id, pattern in patterns:
        for match in re.finditer(pattern, lowered, flags=re.IGNORECASE | re.DOTALL):
            candidates.append((match.start(), section_id, match.group(0).strip()))

    if not candidates:
        return [FilingSection(section_id="full_filing", title="Full Filing", text=text)]

    #SEC filings often repeat item names in a table of contents. Favor later, substantive matches.
    candidates.sort(key=lambda row: row[0])
    deduped: list[tuple[int, str, str]] = []
    seen: set[str] = set()
    for candidate in reversed(candidates):
        if candidate[1] not in seen:
            deduped.append(candidate)
            seen.add(candidate[1])
    deduped.sort(key=lambda row: row[0])

    sections: list[FilingSection] = []
    for idx, (start, section_id, title) in enumerate(deduped):
        end = deduped[idx + 1][0] if idx + 1 < len(deduped) else len(text)
        section_text = text[start:end].strip()
        if len(section_text.split()) >= 40:
            sections.append(
                FilingSection(section_id=section_id, title=title.title(), text=section_text)
            )
    return sections or [FilingSection(section_id="full_filing", title="Full Filing", text=text)]
