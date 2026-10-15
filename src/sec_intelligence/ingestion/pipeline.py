#Raw SEC ingestion pipeline for company metadata, filings, and XBRL Company Facts.
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sec_intelligence.config import PROJECT_ROOT, load_settings
from sec_intelligence.ingestion.cache import FileCache
from sec_intelligence.ingestion.sec_client import SecClient


def ingest_company_raw(
    ticker: str,
    settings: dict[str, Any] | None = None,
    forms: tuple[str, ...] | None = None,
    max_per_form: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    settings = settings or load_settings()
    sec_settings = settings["sec"]
    ticker = ticker.upper()
    selected_forms = forms or tuple(sec_settings["forms"])
    selected_max = max_per_form or int(sec_settings["max_filings_per_form"])

    client = SecClient(
        user_agent=sec_settings["user_agent"],
        max_requests_per_second=float(sec_settings["max_requests_per_second"]),
    )
    cache = FileCache(PROJECT_ROOT / "data" / "raw" / ticker)

    cik, company_name = client.resolve_ticker(ticker)

    submissions = _cached_json(
        cache=cache,
        relative_path="submissions.json",
        loader=lambda: client.submissions(cik),
        force=force,
    )
    company_facts = _cached_json(
        cache=cache,
        relative_path="companyfacts.json",
        loader=lambda: client.company_facts(cik),
        force=force,
    )

    filings = client.filings_from_submissions(
        ticker=ticker,
        cik=cik,
        company_name=company_name,
        submissions=submissions,
        forms=selected_forms,
        max_per_form=selected_max,
    )

    downloaded_files: list[dict[str, str]] = []
    for filing in filings:
        relative_path = f"filings/{filing.accession_number}.html"
        if force or not cache.exists(relative_path):
            cache.write_text(relative_path, client.filing_html(filing))
        downloaded_files.append(
            {
                "accession_number": filing.accession_number,
                "form": filing.form,
                "filing_date": filing.filing_date.isoformat(),
                "report_date": filing.report_date.isoformat() if filing.report_date else "",
                "source_url": filing.filing_url,
                "local_path": relative_path,
            }
        )

    form_counts = Counter(filing.form for filing in filings)
    manifest = {
        "ticker": ticker,
        "company_name": company_name,
        "cik": cik,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "forms_requested": list(selected_forms),
        "max_filings_per_form": selected_max,
        "filing_count": len(filings),
        "filing_count_by_form": dict(form_counts),
        "submissions_path": "submissions.json",
        "company_facts_path": "companyfacts.json",
        "company_facts_entity_name": company_facts.get("entityName", company_name),
        "filings": downloaded_files,
    }
    manifest_path = cache.write_json("manifest.json", manifest)

    return {
        "ticker": ticker,
        "company_name": company_name,
        "cik": cik,
        "filing_count": len(filings),
        "filing_count_by_form": dict(form_counts),
        "raw_root": cache.root,
        "manifest": manifest_path,
    }


def _cached_json(
    cache: FileCache,
    relative_path: str,
    loader: Any,
    force: bool,
) -> Any:
    if not force and cache.exists(relative_path):
        return cache.read_json(relative_path)
    payload = loader()
    cache.write_json(relative_path, payload)
    return payload
