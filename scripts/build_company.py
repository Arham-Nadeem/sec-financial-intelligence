#CLI entry point for downloading and processing a company's SEC filings and XBRL facts.
from __future__ import annotations

import argparse

from sec_intelligence.pipeline import build_company_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ticker", help="Public company ticker, such as AAPL or MSFT")
    args = parser.parse_args()
    paths = build_company_dataset(args.ticker.upper())
    for label, path in paths.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
