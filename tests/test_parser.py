#Unit tests for HTML cleanup and filing section detection.
from sec_intelligence.parsing.filing_parser import html_to_text, split_into_sections


def test_html_to_text_removes_script_content() -> None:
    html = "<html><script>bad()</script><body><h1>Item 1. Business</h1><p>Useful text</p></body></html>"
    text = html_to_text(html)
    assert "bad()" not in text
    assert "Useful text" in text


def test_split_into_sections_finds_10k_items_and_skips_toc_duplicates() -> None:
    text = "\n".join(
        [
            "TABLE OF CONTENTS",
            "Item 1. Business",
            "Item 1A. Risk Factors",
            "Item 7. Management's Discussion and Analysis",
            "Item 1. Business",
            "Business overview. " + "business operations and products. " * 70,
            "Item 1A. Risk Factors",
            "Risk overview. " + "supply chain competition regulation risk. " * 70,
            "Item 7. Management's Discussion and Analysis",
            "Management discussion. " + "revenue margin demand operations. " * 70,
            "Item 7A. Quantitative and Qualitative Disclosures About Market Risk",
            "Market risk. " + "interest rates foreign currency exposure. " * 70,
        ]
    )
    sections = split_into_sections(text, "10-K")
    by_id = {section.section_id: section for section in sections}

    assert {"item_1", "item_1a", "item_7", "item_7a"}.issubset(by_id)
    assert len(by_id["item_1"].text.split()) > 100
    assert "Business overview" in by_id["item_1"].text


def test_split_into_sections_tracks_10q_part_context() -> None:
    text = "\n".join(
        [
            "PART I",
            "Item 1. Financial Statements",
            "financial statements. " * 80,
            "Item 2. Management's Discussion and Analysis",
            "management discussion. " * 80,
            "PART II",
            "Item 1. Legal Proceedings",
            "legal proceedings. " * 80,
            "Item 1A. Risk Factors",
            "risk factors. " * 80,
            "Item 6. Exhibits",
            "exhibits. " * 80,
        ]
    )
    sections = split_into_sections(text, "10-Q")
    ids = {section.section_id for section in sections}

    assert "part_i_item_1" in ids
    assert "part_i_item_2" in ids
    assert "part_ii_item_1" in ids
    assert "part_ii_item_1a" in ids
    assert "part_ii_item_6" in ids
