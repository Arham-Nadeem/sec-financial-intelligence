#Normalize SEC Company Facts into comparable financial metrics and trend calculations.
from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from sec_intelligence.models import FinancialPoint


def _concept_units(company_facts: dict[str, Any], concepts: list[str]) -> tuple[str, dict[str, Any]] | None:
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
    found = _concept_units(company_facts, concepts)
    if not found:
        return []
    _, units = found
    preferred_unit = "USD" if "USD" in units else next(iter(units))
    rows: list[FinancialPoint] = []

    for row in units[preferred_unit]:
        if row.get("form") not in forms or "end" not in row or "filed" not in row:
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


def latest_period_frame(
    company_facts: dict[str, Any], concept_map: dict[str, list[str]]
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for metric, concepts in concept_map.items():
        for point in extract_metric(company_facts, metric, concepts):
            records.append(point.model_dump())
    if not records:
        return pd.DataFrame()

    frame = pd.DataFrame(records)
    frame["filed"] = pd.to_datetime(frame["filed"])
    frame["end"] = pd.to_datetime(frame["end"])
    frame = frame.sort_values(["metric", "end", "filed"]).drop_duplicates(
        subset=["metric", "end", "form", "fiscal_period"], keep="last"
    )
    return frame.reset_index(drop=True)


def annual_kpis(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    annual = frame[(frame["form"] == "10-K") & (frame["fiscal_period"] == "FY")].copy()
    annual = annual.sort_values(["metric", "end"])
    annual["yoy_change_pct"] = annual.groupby("metric")["value"].pct_change() * 100.0
    pivot = annual.pivot_table(index="end", columns="metric", values="value", aggfunc="last")
    if {"operating_cash_flow", "capex"}.issubset(pivot.columns):
        pivot["free_cash_flow"] = pivot["operating_cash_flow"] - pivot["capex"].abs()
    return pivot.reset_index().sort_values("end", ascending=False)
