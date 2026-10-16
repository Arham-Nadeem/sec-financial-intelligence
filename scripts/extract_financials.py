#Build normalized annual KPI and growth tables from cached SEC Company Facts.
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sec_intelligence.analytics.financials import build_financial_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract structured SEC financial analytics.")
    parser.add_argument("ticker", help="Stock ticker whose cached Company Facts should be analyzed.")
    args = parser.parse_args()

    ticker = args.ticker.upper()
    root = Path(__file__).resolve().parents[1]
    raw_path = root / "data" / "raw" / ticker / "companyfacts.json"
    output_dir = root / "data" / "processed" / ticker

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Company Facts not found for {ticker}. Run scripts/ingest_sec.py first."
        )

    company_facts = json.loads(raw_path.read_text(encoding="utf-8"))
    facts, kpis, growth = build_financial_outputs(company_facts)
    output_dir.mkdir(parents=True, exist_ok=True)

    facts_path = output_dir / "financial_facts.csv"
    kpis_path = output_dir / "annual_kpis.csv"
    growth_path = output_dir / "annual_growth.csv"
    facts.to_csv(facts_path, index=False)
    kpis.to_csv(kpis_path, index=False)
    growth.to_csv(growth_path, index=False)

    print(f"Ticker: {ticker}")
    print(f"Normalized facts: {len(facts)}")
    print(f"Annual periods: {len(kpis)}")
    print(f"Metrics: {', '.join(column for column in kpis.columns if column != 'end')}")
    print(f"Financial facts: {facts_path}")
    print(f"Annual KPIs: {kpis_path}")
    print(f"Annual growth: {growth_path}")

    if not kpis.empty:
        print("\nLatest annual KPI rows:")
        print(kpis.head(5).to_string(index=False))
    if not growth.empty:
        print("\nLatest YoY changes:")
        print(growth.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
