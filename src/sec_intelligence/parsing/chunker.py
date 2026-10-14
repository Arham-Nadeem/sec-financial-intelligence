#Create overlapping, section-aware chunks while preserving source metadata.
from __future__ import annotations

import hashlib

from sec_intelligence.models import Chunk, FilingMetadata, FilingSection


def chunk_sections(
    filing: FilingMetadata,
    sections: list[FilingSection],
    chunk_size_words: int = 220,
    overlap_words: int = 40,
) -> list[Chunk]:
    if overlap_words >= chunk_size_words:
        raise ValueError("overlap_words must be smaller than chunk_size_words")

    chunks: list[Chunk] = []
    step = chunk_size_words - overlap_words
    for section in sections:
        words = section.text.split()
        for start in range(0, len(words), step):
            part = words[start : start + chunk_size_words]
            if len(part) < 35:
                continue
            text = " ".join(part)
            stable_key = (
                f"{filing.accession_number}|{section.section_id}|{start}|{text[:80]}"
            )
            chunk_id = hashlib.sha1(stable_key.encode("utf-8")).hexdigest()[:16]
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    ticker=filing.ticker,
                    form=filing.form,
                    filing_date=filing.filing_date,
                    accession_number=filing.accession_number,
                    section_id=section.section_id,
                    section_title=section.title,
                    text=text,
                    source_url=filing.filing_url,
                    metadata={"word_start": start},
                )
            )
    return chunks
