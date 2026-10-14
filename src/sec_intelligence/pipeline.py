#End-to-end pipeline for downloading filings, parsing chunks, and extracting XBRL analytics.
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sec_intelligence.analytics.financials import annual_kpis, latest_period_frame
from sec_intelligence.config import PROJECT_ROOT, load_settings
from sec_intelligence.ingestion.cache import FileCache
from sec_intelligence.ingestion.sec_client import SecClient
from sec_intelligence.models import Chunk
from sec_intelligence.parsing.chunker import chunk_sections
from sec_intelligence.parsing.filing_parser import html_to_text, split_into_sections


def build_company_dataset(ticker: str, settings: dict[str, Any] | None = None) -> dict[str, Path]:
    settings = settings or load_settings()
    sec_settings = settings["sec"]
    retrieval_settings = settings["retrieval"]
    client = SecClient(
        user_agent=sec_settings["user_agent"],
        max_requests_per_second=sec_settings["max_requests_per_second"],
    )
    raw_cache = FileCache(PROJECT_ROOT / "data" / "raw" / ticker.upper())
    processed_root = PROJECT_ROOT / "data" / "processed" / ticker.upper()
    processed_root.mkdir(parents=True, exist_ok=True)

    filings = client.list_filings(
        ticker,
        forms=tuple(sec_settings["forms"]),
        max_per_form=int(sec_settings["max_filings_per_form"]),
    )
    all_chunks: list[Chunk] = []
    for filing in filings:
        html = client.filing_html(filing)
        raw_cache.write_text(f"filings/{filing.accession_number}.html", html)
        text = html_to_text(html)
        sections = split_into_sections(text, filing.form)
        all_chunks.extend(
            chunk_sections(
                filing,
                sections,
                chunk_size_words=int(retrieval_settings["chunk_size_words"]),
                overlap_words=int(retrieval_settings["chunk_overlap_words"]),
            )
        )

    chunks_path = processed_root / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8") as handle:
        for chunk in all_chunks:
            handle.write(chunk.model_dump_json() + "\n")

    cik, _ = client.resolve_ticker(ticker)
    facts = client.company_facts(cik)
    raw_cache.write_json("companyfacts.json", facts)
    frame = latest_period_frame(facts, settings["analytics"]["concepts"])
    facts_path = processed_root / "financial_facts.csv"
    frame.to_csv(facts_path, index=False)
    kpis_path = processed_root / "annual_kpis.csv"
    annual_kpis(frame).to_csv(kpis_path, index=False)

    metadata_path = processed_root / "filings.json"
    metadata_path.write_text(
        json.dumps([row.model_dump(mode="json") for row in filings], indent=2),
        encoding="utf-8",
    )
    return {
        "chunks": chunks_path,
        "financial_facts": facts_path,
        "annual_kpis": kpis_path,
        "filings": metadata_path,
    }


def load_chunks(path: str | Path) -> list[Chunk]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(Chunk.model_validate_json(line))
    return rows
