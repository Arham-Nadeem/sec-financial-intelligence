#SEC EDGAR client for ticker lookup, filing discovery, documents, and XBRL facts.
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date
from typing import Any

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from sec_intelligence.models import FilingMetadata


@dataclass
class SecClient:
    user_agent: str
    max_requests_per_second: float = 5.0

    def __post_init__(self) -> None:
        if "@" not in self.user_agent:
            raise ValueError("SEC user_agent should identify you and include a contact email.")
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": self.user_agent,
                "Accept-Encoding": "gzip, deflate",
            }
        )
        self._last_request = 0.0

    def _throttle(self) -> None:
        min_interval = 1.0 / max(self.max_requests_per_second, 0.1)
        elapsed = time.monotonic() - self._last_request
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=0.5, max=8))
    def _get(self, url: str) -> requests.Response:
        self._throttle()
        response = self._session.get(url, timeout=30)
        self._last_request = time.monotonic()
        response.raise_for_status()
        return response

    def ticker_map(self) -> dict[str, dict[str, Any]]:
        data = self._get("https://www.sec.gov/files/company_tickers.json").json()
        return {row["ticker"].upper(): row for row in data.values()}

    def resolve_ticker(self, ticker: str) -> tuple[str, str]:
        row = self.ticker_map().get(ticker.upper())
        if not row:
            raise ValueError(f"Ticker not found in SEC company ticker mapping: {ticker}")
        return str(row["cik_str"]).zfill(10), row["title"]

    def submissions(self, cik: str) -> dict[str, Any]:
        return self._get(f"https://data.sec.gov/submissions/CIK{cik}.json").json()

    def company_facts(self, cik: str) -> dict[str, Any]:
        return self._get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json").json()

    def list_filings(
        self,
        ticker: str,
        forms: tuple[str, ...] = ("10-K", "10-Q"),
        max_per_form: int = 6,
    ) -> list[FilingMetadata]:
        cik, company_name = self.resolve_ticker(ticker)
        submissions = self.submissions(cik)
        recent = submissions["filings"]["recent"]
        counters = {form: 0 for form in forms}
        filings: list[FilingMetadata] = []

        for idx, form in enumerate(recent["form"]):
            if form not in counters or counters[form] >= max_per_form:
                continue
            accession = recent["accessionNumber"][idx]
            accession_compact = accession.replace("-", "")
            primary_document = recent["primaryDocument"][idx]
            cik_unpadded = str(int(cik))
            url = (
                f"https://www.sec.gov/Archives/edgar/data/{cik_unpadded}/"
                f"{accession_compact}/{primary_document}"
            )
            report_date_raw = recent["reportDate"][idx]
            filings.append(
                FilingMetadata(
                    ticker=ticker.upper(),
                    cik=cik,
                    company_name=company_name,
                    form=form,
                    accession_number=accession,
                    filing_date=date.fromisoformat(recent["filingDate"][idx]),
                    report_date=date.fromisoformat(report_date_raw) if report_date_raw else None,
                    primary_document=primary_document,
                    filing_url=url,
                )
            )
            counters[form] += 1
            if all(count >= max_per_form for count in counters.values()):
                break
        return filings

    def filing_html(self, filing: FilingMetadata) -> str:
        return self._get(filing.filing_url).text
