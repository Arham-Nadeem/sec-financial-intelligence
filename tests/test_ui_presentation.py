#Test presentation helpers used by the Streamlit interface.
from sec_intelligence.ui.presentation import (
    available_metrics,
    format_currency,
    format_percent,
    strategy_display_name,
)


def test_format_currency_uses_compact_financial_units() -> None:
    assert format_currency(416_161_000_000) == "$416.2B"
    assert format_currency(9_447_000_000) == "$9.4B"
    assert format_currency(None) == "—"


def test_format_percent_preserves_direction() -> None:
    assert format_percent(6.4255) == "+6.4%"
    assert format_percent(-9.2273) == "-9.2%"


def test_available_metrics_uses_consistent_order() -> None:
    columns = ["end", "free_cash_flow", "revenue", "net_income"]
    assert available_metrics(columns) == ["revenue", "net_income", "free_cash_flow"]


def test_strategy_display_name_maps_internal_labels() -> None:
    assert strategy_display_name("hybrid_reranked") == "Hybrid + reranker"
    assert strategy_display_name("bm25") == "BM25"


def test_comparison_similarity_note_flags_effectively_unchanged_language():
    from sec_intelligence.ui.presentation import comparison_similarity_note

    note = comparison_similarity_note([
        "The Company depends on suppliers and contract manufacturers for production.",
        "The Company depends on suppliers and contract manufacturers for production.",
    ])
    assert note is not None


def test_comparison_similarity_note_ignores_distinct_language():
    from sec_intelligence.ui.presentation import comparison_similarity_note

    note = comparison_similarity_note([
        "The Company depends on suppliers and contract manufacturers for production.",
        "Regulatory changes may increase compliance costs across several markets.",
    ])
    assert note is None
