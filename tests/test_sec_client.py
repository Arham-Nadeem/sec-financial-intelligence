#Unit tests for SEC filing metadata construction without live network requests.
from sec_intelligence.ingestion.sec_client import SecClient


def test_filings_from_submissions_filters_forms_and_builds_archive_url() -> None:
    submissions = {
        "filings": {
            "recent": {
                "form": ["8-K", "10-Q", "10-K", "10-Q"],
                "accessionNumber": [
                    "0000320193-26-000001",
                    "0000320193-26-000002",
                    "0000320193-25-000003",
                    "0000320193-25-000004",
                ],
                "primaryDocument": ["a8k.htm", "a10q.htm", "a10k.htm", "old10q.htm"],
                "filingDate": ["2026-08-01", "2026-07-31", "2025-10-31", "2025-05-01"],
                "reportDate": ["2026-07-31", "2026-06-27", "2025-09-27", "2025-03-29"],
            }
        }
    }

    filings = SecClient.filings_from_submissions(
        ticker="AAPL",
        cik="0000320193",
        company_name="Apple Inc.",
        submissions=submissions,
        forms=("10-K", "10-Q"),
        max_per_form=1,
    )

    assert [filing.form for filing in filings] == ["10-Q", "10-K"]
    assert filings[0].ticker == "AAPL"
    assert filings[0].cik == "0000320193"
    assert filings[0].filing_url.endswith(
        "/320193/000032019326000002/a10q.htm"
    )


def test_sec_client_rejects_placeholder_user_agent() -> None:
    try:
        SecClient("Your Name your.email@example.com")
    except ValueError as exc:
        assert "SEC_USER_AGENT" in str(exc)
    else:
        raise AssertionError("Expected placeholder SEC user agent to be rejected")
