#Streamlit interface for grounded SEC research, financial trends, evidence, and evaluation.
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from sec_intelligence.answering.grounded import (
    build_grounded_answer,
    compare_filings,
    validate_citations,
)
from sec_intelligence.config import PROJECT_ROOT
from sec_intelligence.ui.presentation import (
    METRIC_LABELS,
    available_metrics,
    format_currency,
    format_percent,
    strategy_display_name,
    EVALUATION_METRIC_LABELS,
    comparison_similarity_note,
)
from sec_intelligence.ui.services import (
    artifact_status,
    build_hybrid_retriever,
    build_reranker,
    company_paths,
    load_evaluation_tables,
    load_financial_tables,
    search_research,
)


st.set_page_config(
    page_title="SEC Financial Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("SEC Financial Intelligence")
st.caption(
    "Source-grounded research over SEC 10-K/10-Q filings with XBRL financial analytics, "
    "hybrid retrieval, local embeddings, and cross-encoder reranking."
)

with st.sidebar:
    st.header("Research workspace")
    ticker = st.text_input("Ticker", value="AAPL", max_chars=8).upper().strip()
    st.caption("Generated filing data stays local and is excluded from Git.")

paths = company_paths(PROJECT_ROOT, ticker)
status = artifact_status(paths)

with st.sidebar:
    st.subheader("System status")
    for label, ready in status.items():
        st.markdown(f"{'✅' if ready else '⚪'} {label}")

    if not status["Parsed filings"]:
        st.info(f"Run `python scripts/ingest_sec.py {ticker} --max-per-form 2`, then parse filings.")
    elif not status["Vector index"]:
        st.info(f"Run `python scripts/build_vector_index.py {ticker}` to enable semantic search.")

if not status["Parsed filings"]:
    st.warning(f"No parsed filing dataset found for {ticker}.")
    st.stop()

if not status["Vector index"]:
    st.warning(f"No vector index found for {ticker}. Build it before launching the research interface.")
    st.stop()


@st.cache_resource(show_spinner="Loading local embedding model...")
def get_hybrid_retriever(ticker_value: str):
    #Ticker is the cache key so switching companies loads the correct local index.
    return build_hybrid_retriever(company_paths(PROJECT_ROOT, ticker_value))


@st.cache_resource(show_spinner="Loading local cross-encoder reranker...")
def get_reranker():
    #Load the slower reranker only when a research/search action actually requests it.
    return build_reranker()

research_tab, financial_tab, evidence_tab, evaluation_tab = st.tabs(
    ["Research Assistant", "Financial Trends", "Evidence Explorer", "Retrieval Evaluation"]
)

with research_tab:
    st.subheader("Grounded filing research")
    st.write(
        "Ask a question about management disclosures. The answer is extractive: every displayed "
        "claim maps directly to retrieved SEC evidence rather than a model-generated fact."
    )

    left, right = st.columns([3, 1])
    with left:
        question = st.text_area(
            "Research question",
            value="How does Apple depend on outside companies for manufacturing?",
            height=105,
        )
    with right:
        mode = st.radio("Mode", ["Grounded answer", "Compare filings"], index=0)
        evidence_limit = st.slider("Max evidence", min_value=1, max_value=6, value=4)

    if st.button("Run research", type="primary", use_container_width=False):
        if not question.strip():
            st.warning("Enter a research question.")
        else:
            #Keep expensive retrieval work behind an explicit user action.
            with st.spinner("Searching filings and reranking evidence..."):
                hybrid = get_hybrid_retriever(ticker)
                reranker = get_reranker()
                results = search_research(
                    question,
                    hybrid,
                    reranker,
                    top_k=max(10, evidence_limit * 3),
                    candidate_k=30,
                )
                if mode == "Compare filings":
                    grounded = compare_filings(question, results, max_filings=evidence_limit)
                else:
                    grounded = build_grounded_answer(question, results, max_evidence=evidence_limit)
                validation = validate_citations(grounded)

            metric_a, metric_b, metric_c = st.columns(3)
            metric_a.metric("Evidence passages", grounded.evidence_count)
            metric_b.metric("Citation precision", f"{validation['citation_precision']:.0%}")
            metric_c.metric("Citation coverage", f"{validation['citation_coverage']:.0%}")

            st.markdown("#### Answer")
            if mode == "Compare filings":
                note = comparison_similarity_note([citation.excerpt for citation in grounded.citations])
                if note:
                    st.success(note)
                for citation in grounded.citations:
                    st.markdown(f"**{citation.filing_date} · {citation.form} · {citation.section_title}**")
                    st.info(f"{citation.excerpt} [{citation.citation_id}]")
            else:
                st.info(grounded.answer)

            st.markdown("#### SEC evidence")
            for citation in grounded.citations:
                label = (
                    f"[{citation.citation_id}] {citation.form} · {citation.filing_date} · "
                    f"{citation.section_title}"
                )
                with st.expander(label, expanded=citation.citation_id == 1):
                    st.write(citation.excerpt)
                    st.caption(f"Chunk ID: {citation.chunk_id}")
                    st.markdown(f"[Open SEC filing]({citation.source_url})")

with financial_tab:
    st.subheader("Structured financial trends")
    kpis, growth = load_financial_tables(paths)

    if kpis.empty:
        st.info(f"No KPI table found. Run `python scripts/extract_financials.py {ticker}`.")
    else:
        kpis = kpis.copy()
        kpis["end"] = pd.to_datetime(kpis["end"])
        kpis = kpis.sort_values("end", ascending=False)
        if not growth.empty:
            growth = growth.copy()
            growth["end"] = pd.to_datetime(growth["end"])
            growth = growth.sort_values("end", ascending=False)

        metrics = available_metrics(kpis.columns)
        latest = kpis.iloc[0]
        latest_growth = growth.iloc[0] if not growth.empty else None

        st.caption(f"Latest annual period: {latest['end'].date()}")
        card_columns = st.columns(3)
        for index, metric in enumerate(metrics[:6]):
            delta_column = f"{metric}_yoy_pct"
            delta = None
            if latest_growth is not None and delta_column in growth.columns:
                delta = format_percent(latest_growth[delta_column])
            card_columns[index % 3].metric(
                METRIC_LABELS[metric],
                format_currency(latest[metric]),
                delta=delta,
            )

        st.markdown("#### Trend explorer")
        selected_metrics = st.multiselect(
            "Metrics",
            metrics,
            default=[metric for metric in ["revenue", "net_income", "free_cash_flow"] if metric in metrics],
            format_func=lambda metric: METRIC_LABELS.get(metric, metric),
        )
        if selected_metrics:
            chart = kpis.sort_values("end")[["end", *selected_metrics]].copy()
            #Scale large financial values so the chart is recruiter-readable at a glance.
            for metric in selected_metrics:
                chart[metric] = chart[metric] / 1_000_000_000
            chart = chart.set_index("end").rename(
                columns={metric: f"{METRIC_LABELS[metric]} ($B)" for metric in selected_metrics}
            )
            st.caption("USD billions")
            st.line_chart(chart, use_container_width=True)

        with st.expander("View annual KPI table"):
            display = kpis.copy()
            display["end"] = display["end"].dt.date
            st.dataframe(display, use_container_width=True, hide_index=True)

        if not growth.empty:
            with st.expander("View year-over-year growth table"):
                display_growth = growth.copy()
                display_growth["end"] = display_growth["end"].dt.date
                st.dataframe(display_growth, use_container_width=True, hide_index=True)

with evidence_tab:
    st.subheader("Evidence explorer")
    st.write("Inspect the ranked passages behind the research layer and compare retrieval signals.")

    query = st.text_input(
        "Evidence search",
        value="supplier manufacturing disruption risk",
        key="evidence_query",
    )
    control_a, control_b, control_c = st.columns(3)
    with control_a:
        form_filter = st.selectbox("Form", ["All", "10-K", "10-Q"])
    with control_b:
        result_count = st.slider("Results", min_value=3, max_value=12, value=6)
    with control_c:
        rerank_enabled = st.checkbox("Cross-encoder reranking", value=True)

    run_evidence_search = st.button("Search evidence", key="run_evidence_search")
    if run_evidence_search and query.strip():
        hybrid = get_hybrid_retriever(ticker)
        form = None if form_filter == "All" else form_filter
        candidates = hybrid.search(query, top_k=max(24, result_count), candidate_k=30, form=form)
        results = (
            get_reranker().rerank(query, candidates, top_k=result_count)
            if rerank_enabled
            else candidates[:result_count]
        )

        for rank, result in enumerate(results, start=1):
            chunk = result.chunk
            st.markdown(
                f"**#{rank} · {chunk.form} · {chunk.filing_date} · {chunk.section_title}**"
            )
            score_columns = st.columns(4)
            score_columns[0].metric("Hybrid", f"{result.score:.4f}")
            score_columns[1].metric("BM25", f"{result.bm25_score:.3f}")
            score_columns[2].metric("Vector", f"{result.vector_score:.3f}")
            score_columns[3].metric(
                "Rerank",
                "—" if result.rerank_score is None else f"{result.rerank_score:.3f}",
            )
            st.write(chunk.text)
            st.markdown(f"[Open SEC filing]({chunk.source_url})")
            st.divider()

with evaluation_tab:
    st.subheader("Retrieval evaluation")
    st.write(
        "Benchmark results compare lexical, semantic, hybrid, and reranked retrieval using labeled "
        "SEC research questions rather than relying on anecdotal search examples."
    )

    summary, details = load_evaluation_tables(paths)
    if summary.empty:
        st.info(f"No benchmark output found. Run `python scripts/evaluate_retrieval.py {ticker}`.")
    else:
        summary = summary.copy()
        summary["strategy_label"] = summary["strategy"].map(strategy_display_name)
        metric_columns = [
            column
            for column in ["recall@5", "mrr", "ndcg@5", "relevant_rate@5", "keyword_hit@5"]
            if column in summary.columns
        ]

        best_recall = summary.loc[summary["recall@5"].idxmax()] if "recall@5" in summary else None
        best_mrr = summary.loc[summary["mrr"].idxmax()] if "mrr" in summary else None
        best_ndcg = summary.loc[summary["ndcg@5"].idxmax()] if "ndcg@5" in summary else None

        best_columns = st.columns(3)
        if best_recall is not None:
            best_columns[0].metric("Best Recall@5", f"{best_recall['recall@5']:.3f}")
            best_columns[0].caption(strategy_display_name(str(best_recall["strategy"])))
        if best_mrr is not None:
            best_columns[1].metric("Best MRR", f"{best_mrr['mrr']:.3f}")
            best_columns[1].caption(strategy_display_name(str(best_mrr["strategy"])))
        if best_ndcg is not None:
            best_columns[2].metric("Best nDCG@5", f"{best_ndcg['ndcg@5']:.3f}")
            best_columns[2].caption(strategy_display_name(str(best_ndcg["strategy"])))

        st.markdown("#### Strategy comparison")
        selected_metric = st.selectbox(
            "Evaluation metric",
            metric_columns,
            index=0,
            format_func=lambda metric: EVALUATION_METRIC_LABELS.get(metric, metric),
        )
        comparison = (
            summary[["strategy_label", selected_metric]]
            .set_index("strategy_label")
            .rename(columns={selected_metric: EVALUATION_METRIC_LABELS.get(selected_metric, selected_metric)})
        )
        st.bar_chart(comparison, use_container_width=True)
        st.caption("Each metric is shown independently; retrieval metrics are not additive.")

        display_summary = summary.drop(columns=["strategy_label"])
        display_summary["strategy"] = display_summary["strategy"].map(strategy_display_name)
        st.dataframe(display_summary, use_container_width=True, hide_index=True)

        if not details.empty:
            with st.expander("Question-level benchmark details"):
                st.dataframe(details, use_container_width=True, hide_index=True)
