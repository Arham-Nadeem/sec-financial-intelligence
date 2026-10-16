#Normalize SEC Company Facts into annual financial KPIs and year-over-year trends.
from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from sec_intelligence.models import FinancialPoint


DEFAULT_CONCEPTS: dict[str, list[str]] = {
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
    ],
    "operating_income": ["OperatingIncomeLoss"],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "operating_cash_flow": ["NetCashProvidedByUsedInOperatingActivities"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment"],
}


def _concept_units(
    company_facts: dict[str, Any], concepts: list[str]
) -> tuple[str, dict[str, Any]] | None:
    """Return the first available US-GAAP concept from a preference list."""
    us_gaap = company_facts.get("facts", {}).get("us-gaap", {})
    for concept in concepts:
        payload = us_gaap.get(concept)
        if payload and payload.get("units"):
            return concept, payload["units"]
    return None


def extract_metric(
    company_facts: dict[str, Any],
    metric: str,
    concepts: list[str],
    forms: tuple[str, ...] = ("10-K", "10-Q"),
) -> list[FinancialPoint]:
    """Extract one normalized metric from SEC Company Facts."""
    found = _concept_units(company_facts, concepts)
    if not found:
        return []

    _, units = found
    preferred_unit = "USD" if "USD" in units else next(iter(units))
    rows: list[FinancialPoint] = []

    for row in units[preferred_unit]:
        if row.get("form") not in forms or "end" not in row or "filed" not in row:
            continue
        if row.get("val") is None:
            continue
        rows.append(
            FinancialPoint(
                metric=metric,
                fiscal_year=row.get("fy"),
                fiscal_period=row.get("fp"),
                form=row["form"],
                filed=date.fromisoformat(row["filed"]),
                start=date.fromisoformat(row["start"]) if row.get("start") else None,
                end=date.fromisoformat(row["end"]),
                value=float(row["val"]),
                unit=preferred_unit,
                accession_number=row.get("accn"),
            )
        )
    return rows


def financial_facts_frame(
    company_facts: dict[str, Any],
    concept_map: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Create a deduplicated long-form table of normalized SEC financial facts."""
    records: list[dict[str, Any]] = []
    for metric, concepts in (concept_map or DEFAULT_CONCEPTS).items():
        for point in extract_metric(company_facts, metric, concepts):
            records.append(point.model_dump())

    if not records:
        return pd.DataFrame(
            columns=[
                "metric",
                "fiscal_year",
                "fiscal_period",
                "form",
                "filed",
                "start",
                "end",
                "value",
                "unit",
                "accession_number",
            ]
        )

    frame = pd.DataFrame(records)
    frame["filed"] = pd.to_datetime(frame["filed"])
    frame["start"] = pd.to_datetime(frame["start"])
    frame["end"] = pd.to_datetime(frame["end"])
    frame = frame.sort_values(["metric", "end", "filed"])

    #Keep the most recently filed version of each economic period.
    frame = frame.drop_duplicates(
        subset=["metric", "start", "end", "form", "fiscal_period"],
        keep="last",
    )
    return frame.reset_index(drop=True)


def _annual_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Select full-year 10-K observations while excluding quarterly-duration facts."""
    if frame.empty:
        return frame.copy()

    annual = frame[(frame["form"] == "10-K") & (frame["fiscal_period"] == "FY")].copy()
    if annual.empty:
        return annual

    duration_days = (annual["end"] - annual["start"]).dt.days
    has_start = annual["start"].notna()
    annual = annual[(~has_start) | duration_days.between(300, 400)]

    #Comparative prior-year values can be repeated in later 10-Ks. Keep the newest filing.
    annual = annual.sort_values(["metric", "end", "filed"]).drop_duplicates(
        subset=["metric", "end"], keep="last"
    )
    return annual.reset_index(drop=True)


def annual_kpis(frame: pd.DataFrame) -> pd.DataFrame:
    """Pivot annual facts into one row per fiscal period and calculate free cash flow."""
    annual = _annual_rows(frame)
    if annual.empty:
        return pd.DataFrame()

    pivot = annual.pivot_table(index="end", columns="metric", values="value", aggfunc="last")
    pivot = pivot.sort_index()
    if {"operating_cash_flow", "capex"}.issubset(pivot.columns):
        pivot["free_cash_flow"] = pivot["operating_cash_flow"] - pivot["capex"].abs()

    result = pivot.reset_index()
    result.columns.name = None
    return result.sort_values("end", ascending=False).reset_index(drop=True)


def annual_growth(kpis: pd.DataFrame) -> pd.DataFrame:
    """Calculate year-over-year percentage changes for each annual KPI."""
    if kpis.empty or "end" not in kpis.columns:
        return pd.DataFrame()

    ordered = kpis.sort_values("end").copy()
    metric_columns = [column for column in ordered.columns if column != "end"]
    growth = ordered[["end"]].copy()
    for column in metric_columns:
        growth[f"{column}_yoy_pct"] = ordered[column].pct_change(fill_method=None) * 100.0
    return growth.sort_values("end", ascending=False).reset_index(drop=True)


def build_financial_outputs(
    company_facts: dict[str, Any],
    concept_map: dict[str, list[str]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build long-form facts, annual KPIs, and YoY growth tables in one pass."""
    facts = financial_facts_frame(company_facts, concept_map)
    kpis = annual_kpis(facts)
    growth = annual_growth(kpis)
    return facts, kpis, growth
