#Unit tests for HTML cleanup and filing section detection.
from sec_intelligence.parsing.filing_parser import html_to_text, split_into_sections


def test_html_to_text_removes_script_content() -> None:
    html = "<html><script>bad()</script><body><h1>Item 1. Business</h1><p>Useful text</p></body></html>"
    text = html_to_text(html)
    assert "bad()" not in text
    assert "Useful text" in text


def test_split_into_sections_finds_10k_items() -> None:
    text = (
        "Item 1. Business " + "business " * 80
        + " Item 1A. Risk Factors " + "risk " * 80
        + " Item 7. Management's Discussion " + "management " * 80
    )
    sections = split_into_sections(text, "10-K")
    ids = {section.section_id for section in sections}
    assert {"item_1", "item_1a", "item_7"}.issubset(ids)
