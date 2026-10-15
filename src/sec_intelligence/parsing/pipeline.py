#Process cached SEC filing HTML into section and chunk datasets.
from __future__ import annotations

import json
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

from sec_intelligence.config import PROJECT_ROOT, load_settings
from sec_intelligence.models import FilingMetadata
from sec_intelligence.parsing.chunker import chunk_sections
from sec_intelligence.parsing.filing_parser import html_to_text, split_into_sections


def parse_cached_filings(
    ticker: str,
    settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    settings = settings or load_settings()
    ticker = ticker.upper()
    raw_root = PROJECT_ROOT / "data" / "raw" / ticker
    manifest_path = raw_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No raw manifest found for {ticker}. Run scripts/ingest_sec.py {ticker} first."
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    processed_root = PROJECT_ROOT / "data" / "processed" / ticker
    processed_root.mkdir(parents=True, exist_ok=True)

    chunk_size = int(settings["retrieval"]["chunk_size_words"])
    overlap = int(settings["retrieval"]["chunk_overlap_words"])

    section_rows: list[dict[str, Any]] = []
    chunk_rows: list[str] = []
    section_counts: Counter[str] = Counter()
    filing_summaries: list[dict[str, Any]] = []

    for filing_row in manifest.get("filings", []):
        local_path = raw_root / filing_row["local_path"]
        if not local_path.exists():
            raise FileNotFoundError(f"Cached filing is missing: {local_path}")

        filing = _metadata_from_manifest(manifest, filing_row)
        html = local_path.read_text(encoding="utf-8", errors="replace")
        text = html_to_text(html)
        sections = split_into_sections(text, filing.form)
        chunks = chunk_sections(
            filing,
            sections,
            chunk_size_words=chunk_size,
            overlap_words=overlap,
        )

        for section in sections:
            section_counts[section.section_id] += 1
            section_rows.append(
                {
                    "ticker": filing.ticker,
                    "form": filing.form,
                    "filing_date": filing.filing_date.isoformat(),
                    "accession_number": filing.accession_number,
                    "section_id": section.section_id,
                    "section_title": section.title,
                    "word_count": len(section.text.split()),
                    "source_url": filing.filing_url,
                    "text": section.text,
                }
            )
        chunk_rows.extend(chunk.model_dump_json() for chunk in chunks)
        filing_summaries.append(
            {
                "form": filing.form,
                "filing_date": filing.filing_date.isoformat(),
                "accession_number": filing.accession_number,
                "sections": len(sections),
                "chunks": len(chunks),
                "section_ids": [section.section_id for section in sections],
            }
        )

    sections_path = processed_root / "sections.jsonl"
    with sections_path.open("w", encoding="utf-8") as handle:
        for row in section_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    chunks_path = processed_root / "chunks.jsonl"
    chunks_path.write_text(
        "\n".join(chunk_rows) + ("\n" if chunk_rows else ""),
        encoding="utf-8",
    )

    summary = {
        "ticker": ticker,
        "company_name": manifest.get("company_name", ticker),
        "filings_processed": len(filing_summaries),
        "sections_created": len(section_rows),
        "chunks_created": len(chunk_rows),
        "section_counts": dict(sorted(section_counts.items())),
        "filings": filing_summaries,
    }
    summary_path = processed_root / "parsing_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {
        "ticker": ticker,
        "filings_processed": len(filing_summaries),
        "sections_created": len(section_rows),
        "chunks_created": len(chunk_rows),
        "section_counts": dict(sorted(section_counts.items())),
        "sections_path": sections_path,
        "chunks_path": chunks_path,
        "summary_path": summary_path,
    }


def _metadata_from_manifest(
    manifest: dict[str, Any], filing_row: dict[str, Any]
) -> FilingMetadata:
    accession = filing_row["accession_number"]
    local_path = Path(filing_row.get("local_path", f"filings/{accession}.html"))
    report_date = filing_row.get("report_date") or None
    return FilingMetadata(
        ticker=manifest["ticker"],
        cik=manifest["cik"],
        company_name=manifest.get("company_name", manifest["ticker"]),
        form=filing_row["form"],
        accession_number=accession,
        filing_date=date.fromisoformat(filing_row["filing_date"]),
        report_date=date.fromisoformat(report_date) if report_date else None,
        primary_document=filing_row.get("primary_document", local_path.name),
        filing_url=filing_row["source_url"],
    )
