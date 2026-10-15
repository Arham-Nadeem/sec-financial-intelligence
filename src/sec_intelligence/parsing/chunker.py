#Create overlapping, section-aware chunks while preserving source metadata.
from __future__ import annotations

import hashlib
import re

from sec_intelligence.models import Chunk, FilingMetadata, FilingSection


def chunk_sections(
    filing: FilingMetadata,
    sections: list[FilingSection],
    chunk_size_words: int = 220,
    overlap_words: int = 40,
    minimum_chunk_words: int = 35,
) -> list[Chunk]:
    if chunk_size_words <= 0:
        raise ValueError("chunk_size_words must be positive")
    if overlap_words < 0:
        raise ValueError("overlap_words cannot be negative")
    if overlap_words >= chunk_size_words:
        raise ValueError("overlap_words must be smaller than chunk_size_words")

    chunks: list[Chunk] = []
    for section in sections:
        windows = _sentence_aware_windows(
            section.text,
            chunk_size_words=chunk_size_words,
            overlap_words=overlap_words,
            minimum_chunk_words=minimum_chunk_words,
        )
        for chunk_index, (text, word_start, word_end) in enumerate(windows):
            stable_key = (
                f"{filing.accession_number}|{section.section_id}|{chunk_index}|{text[:120]}"
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
                    metadata={
                        "chunk_index": chunk_index,
                        "word_start": word_start,
                        "word_end": word_end,
                        "word_count": len(text.split()),
                    },
                )
            )
    return chunks


def _sentence_aware_windows(
    text: str,
    chunk_size_words: int,
    overlap_words: int,
    minimum_chunk_words: int,
) -> list[tuple[str, int, int]]:
    sentences = _split_sentences(text)
    if not sentences:
        return []

    windows: list[tuple[str, int, int]] = []
    index = 0
    word_cursor = 0

    while index < len(sentences):
        selected: list[str] = []
        selected_words = 0
        start_index = index
        start_word = word_cursor

        while index < len(sentences):
            sentence_words = len(sentences[index].split())
            if selected and selected_words + sentence_words > chunk_size_words:
                break
            selected.append(sentences[index])
            selected_words += sentence_words
            index += 1
            if selected_words >= chunk_size_words:
                break

        if selected_words < minimum_chunk_words:
            if windows and selected:
                previous_text, previous_start, _ = windows[-1]
                merged = f"{previous_text} {' '.join(selected)}".strip()
                windows[-1] = (merged, previous_start, start_word + selected_words)
            break

        chunk_text = " ".join(selected).strip()
        windows.append((chunk_text, start_word, start_word + selected_words))

        if index >= len(sentences):
            break

        #Back up by complete sentences until roughly the requested overlap is reached.
        overlap_count = 0
        overlap_sentence_count = 0
        for sentence in reversed(selected):
            overlap_count += len(sentence.split())
            overlap_sentence_count += 1
            if overlap_count >= overlap_words:
                break

        next_index = max(start_index + 1, index - overlap_sentence_count)
        advanced_words = sum(len(sentence.split()) for sentence in sentences[start_index:next_index])
        word_cursor = start_word + advanced_words
        index = next_index

    return windows


def _split_sentences(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(])", normalized)
    return [sentence.strip() for sentence in sentences if sentence.strip()]
