#Streamlit research interface for filing search, grounded evidence, and KPI trends.
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from sec_intelligence.answering.grounded import compose_extractive_answer
from sec_intelligence.config import PROJECT_ROOT, load_settings
from sec_intelligence.pipeline import load_chunks
from sec_intelligence.retrieval.hybrid import HybridRetriever


st.set_page_config(page_title="SEC Financial Intelligence", layout="wide")
st.title("SEC Financial Intelligence")
st.caption("Structured XBRL analytics + section-aware hybrid retrieval over 10-K and 10-Q filings")

settings = load_settings()
ticker = st.sidebar.text_input("Ticker", value="AAPL").upper().strip()
processed = PROJECT_ROOT / "data" / "processed" / ticker
chunks_path = processed / "chunks.jsonl"
kpis_path = processed / "annual_kpis.csv"

if not chunks_path.exists():
    st.warning(f"No processed dataset found for {ticker}. Run: python scripts/build_company.py {ticker}")
    st.stop()


@st.cache_resource(show_spinner="Loading local retrieval models...")
def get_retriever(path: str) -> HybridRetriever:
    chunks = load_chunks(path)
    config = settings["retrieval"]
    return HybridRetriever(
        chunks=chunks,
        embedding_model_name=config["embedding_model"],
        reranker_model_name=config["reranker_model"],
        vector_weight=float(config["vector_weight"]),
        bm25_weight=float(config["bm25_weight"]),
        use_reranker=bool(config["use_reranker"]),
    )


tab_research, tab_financials, tab_evidence = st.tabs(
    ["Research Assistant", "Financial Trends", "Evidence Explorer"]
)

with tab_research:
    question = st.text_area(
        "Research question",
        value="How has management described supply-chain risk across recent filings?",
        height=90,
    )
    if st.button("Research", type="primary"):
        retriever = get_retriever(str(chunks_path))
        results = retriever.search(
            question,
            candidate_k=int(settings["retrieval"]["candidate_k"]),
            top_k=int(settings["retrieval"]["top_k"]),
        )
        st.markdown(compose_extractive_answer(question, results))
        st.subheader("Ranked Sources")
        for result in results:
            with st.expander(
                f"{result.chunk.form} · {result.chunk.filing_date} · {result.chunk.section_title}"
            ):
                st.write(result.chunk.text)
                st.caption(
                    f"Hybrid {result.score:.3f} | Vector {result.vector_score:.3f} | "
                    f"BM25 {result.bm25_score:.3f} | Rerank {result.rerank_score}"
                )
                st.link_button("Open SEC source", result.chunk.source_url)

with tab_financials:
    if not kpis_path.exists():
        st.info("No KPI file found for this ticker.")
    else:
        kpis = pd.read_csv(kpis_path)
        st.dataframe(kpis, use_container_width=True)
        numeric = [column for column in kpis.columns if column != "end"]
        if numeric:
            chart = kpis.sort_values("end").set_index("end")[numeric]
            st.line_chart(chart)

with tab_evidence:
    retriever = get_retriever(str(chunks_path))
    query = st.text_input("Search filing evidence", value="artificial intelligence investment")
    top_k = st.slider("Results", 3, 12, 6)
    for result in retriever.search(query, candidate_k=max(20, top_k), top_k=top_k):
        st.markdown(
            f"**{result.chunk.form} · {result.chunk.filing_date} · {result.chunk.section_title}**"
        )
        st.write(result.chunk.text)
        st.divider()
