#Unit tests for SEC Company Facts normalization and annual KPI calculations.
from sec_intelligence.analytics.financials import annual_kpis, latest_period_frame


def test_annual_kpis_calculates_free_cash_flow() -> None:
    facts = {
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {"USD": [{"fy": 2025, "fp": "FY", "form": "10-K", "filed": "2026-02-01", "start": "2025-01-01", "end": "2025-12-31", "val": 1000, "accn": "a"}]}
                },
                "NetCashProvidedByUsedInOperatingActivities": {
                    "units": {"USD": [{"fy": 2025, "fp": "FY", "form": "10-K", "filed": "2026-02-01", "start": "2025-01-01", "end": "2025-12-31", "val": 300, "accn": "a"}]}
                },
                "PaymentsToAcquirePropertyPlantAndEquipment": {
                    "units": {"USD": [{"fy": 2025, "fp": "FY", "form": "10-K", "filed": "2026-02-01", "start": "2025-01-01", "end": "2025-12-31", "val": 75, "accn": "a"}]}
                },
            }
        }
    }
    concepts = {
        "revenue": ["Revenues"],
        "operating_cash_flow": ["NetCashProvidedByUsedInOperatingActivities"],
        "capex": ["PaymentsToAcquirePropertyPlantAndEquipment"],
    }
    frame = latest_period_frame(facts, concepts)
    kpis = annual_kpis(frame)
    assert float(kpis.iloc[0]["free_cash_flow"]) == 225.0
