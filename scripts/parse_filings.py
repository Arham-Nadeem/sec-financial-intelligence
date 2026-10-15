#CLI entry point for converting cached SEC filings into sections and chunks.
from __future__ import annotations

import argparse

from sec_intelligence.parsing.pipeline import parse_cached_filings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse cached 10-K/10-Q filings into section-aware research chunks."
    )
    parser.add_argument("ticker", help="Public company ticker, such as AAPL or MSFT")
    args = parser.parse_args()

    result = parse_cached_filings(args.ticker)
    print(f"Ticker: {result['ticker']}")
    print(f"Filings processed: {result['filings_processed']}")
    print(f"Sections created: {result['sections_created']}")
    print(f"Chunks created: {result['chunks_created']}")
    print("Sections detected:")
    for section_id, count in result["section_counts"].items():
        print(f"  {section_id}: {count}")
    print(f"Sections file: {result['sections_path']}")
    print(f"Chunks file: {result['chunks_path']}")
    print(f"Summary: {result['summary_path']}")


if __name__ == "__main__":
    main()
