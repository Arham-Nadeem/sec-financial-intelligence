#CLI for downloading raw SEC filing documents and structured Company Facts data.
from __future__ import annotations

import argparse

from sec_intelligence.ingestion.pipeline import ingest_company_raw


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and cache SEC filings plus Company Facts for a public company."
    )
    parser.add_argument("ticker", help="Public company ticker, such as AAPL or MSFT")
    parser.add_argument(
        "--forms",
        nargs="+",
        default=None,
        help="SEC forms to download. Defaults to forms in config/settings.yaml.",
    )
    parser.add_argument(
        "--max-per-form",
        type=int,
        default=None,
        help="Maximum filings to download for each requested form.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload cached SEC JSON and filing documents.",
    )
    args = parser.parse_args()

    result = ingest_company_raw(
        ticker=args.ticker,
        forms=tuple(args.forms) if args.forms else None,
        max_per_form=args.max_per_form,
        force=args.force,
    )

    print(f"Company: {result['company_name']} ({result['ticker']})")
    print(f"CIK: {result['cik']}")
    print(f"Filings cached: {result['filing_count']}")
    for form, count in sorted(result["filing_count_by_form"].items()):
        print(f"  {form}: {count}")
    print(f"Raw data: {result['raw_root']}")
    print(f"Manifest: {result['manifest']}")


if __name__ == "__main__":
    main()
