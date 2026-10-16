#Test SEC financial normalization, free-cash-flow calculation, and YoY trend logic.
from __future__ import annotations

import pandas as pd
import pytest

from sec_intelligence.analytics.financials import annual_growth, annual_kpis, financial_facts_frame


def sample_company_facts() -> dict:
    def rows(values: list[tuple[str, str, int, str]]) -> dict:
        return {
            "units": {
                "USD": [
                    {
                        "start": start,
                        "end": end,
                        "val": value,
                        "accn": accn,
                        "fy": int(end[:4]),
                        "fp": "FY",
                        "form": "10-K",
                        "filed": filed,
                    }
                    for start, end, value, accn, filed in [
                        (*values[0], "2024-11-01"),
                        (*values[1], "2025-10-31"),
                    ]
                ]
            }
        }

    return {
        "facts": {
            "us-gaap": {
                "RevenueFromContractWithCustomerExcludingAssessedTax": rows(
                    [
                        ("2023-10-01", "2024-09-28", 1000, "a"),
                        ("2024-09-29", "2025-09-27", 1100, "b"),
                    ]
                ),
                "OperatingIncomeLoss": rows(
                    [
                        ("2023-10-01", "2024-09-28", 250, "a"),
                        ("2024-09-29", "2025-09-27", 300, "b"),
                    ]
                ),
                "NetCashProvidedByUsedInOperatingActivities": rows(
                    [
                        ("2023-10-01", "2024-09-28", 300, "a"),
                        ("2024-09-29", "2025-09-27", 330, "b"),
                    ]
                ),
                "PaymentsToAcquirePropertyPlantAndEquipment": rows(
                    [
                        ("2023-10-01", "2024-09-28", 80, "a"),
                        ("2024-09-29", "2025-09-27", 90, "b"),
                    ]
                ),
            }
        }
    }


def test_financial_facts_frame_extracts_normalized_metrics() -> None:
    frame = financial_facts_frame(sample_company_facts())
    assert set(frame["metric"]) == {
        "revenue",
        "operating_income",
        "operating_cash_flow",
        "capex",
    }
    assert len(frame) == 8


def test_annual_kpis_calculates_free_cash_flow() -> None:
    frame = financial_facts_frame(sample_company_facts())
    kpis = annual_kpis(frame)
    latest = kpis.iloc[0]
    assert latest["revenue"] == 1100
    assert latest["free_cash_flow"] == 240


def test_annual_growth_calculates_yoy_percentages() -> None:
    frame = financial_facts_frame(sample_company_facts())
    growth = annual_growth(annual_kpis(frame))
    latest = growth.iloc[0]
    assert latest["revenue_yoy_pct"] == pytest.approx(10.0)
    assert latest["operating_income_yoy_pct"] == pytest.approx(20.0)
    assert pd.notna(latest["free_cash_flow_yoy_pct"])
