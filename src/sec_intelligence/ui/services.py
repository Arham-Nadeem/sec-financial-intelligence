#Load cached datasets and construct local retrieval components used by the Streamlit app.
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from sec_intelligence.models import RetrievalResult
from sec_intelligence.retrieval.bm25 import BM25Retriever, load_chunks
from sec_intelligence.retrieval.hybrid import HybridRetriever
from sec_intelligence.retrieval.reranker import CrossEncoderReranker
from sec_intelligence.retrieval.vector import load_saved_vector_retriever


@dataclass(frozen=True)
class CompanyPaths:
    """Resolved local artifact paths for one ticker."""

    ticker: str
    chunks: Path
    vector_index: Path
    kpis: Path
    growth: Path
    evaluation_summary: Path
    evaluation_details: Path


def company_paths(root: Path, ticker: str) -> CompanyPaths:
    """Resolve generated artifact locations for a ticker."""
    normalized = ticker.upper().strip()
    processed = root / "data" / "processed" / normalized
    return CompanyPaths(
        ticker=normalized,
        chunks=processed / "chunks.jsonl",
        vector_index=root / "data" / "indexes" / normalized / "vector",
        kpis=processed / "annual_kpis.csv",
        growth=processed / "annual_growth.csv",
        evaluation_summary=processed / "evaluation" / "retrieval_summary.csv",
        evaluation_details=processed / "evaluation" / "retrieval_details.csv",
    )


def artifact_status(paths: CompanyPaths) -> dict[str, bool]:
    """Report whether each generated project artifact is available locally."""
    return {
        "Parsed filings": paths.chunks.exists(),
        "Vector index": (paths.vector_index / "chunks.faiss").exists()
        and (paths.vector_index / "metadata.json").exists(),
        "Financial KPIs": paths.kpis.exists(),
        "Evaluation results": paths.evaluation_summary.exists(),
    }


def build_hybrid_retriever(paths: CompanyPaths) -> HybridRetriever:
    """Construct BM25 and vector retrieval without loading the reranker."""
    chunks = load_chunks(paths.chunks)
    bm25 = BM25Retriever(chunks)
    vector = load_saved_vector_retriever(paths.chunks, paths.vector_index)
    return HybridRetriever(bm25, vector)


def build_reranker() -> CrossEncoderReranker:
    """Construct the local cross-encoder only when reranking is requested."""
    return CrossEncoderReranker()


def search_research(
    query: str,
    hybrid: HybridRetriever,
    reranker: CrossEncoderReranker,
    top_k: int = 8,
    candidate_k: int = 24,
) -> list[RetrievalResult]:
    """Run hybrid retrieval followed by local cross-encoder reranking."""
    candidates = hybrid.search(
        query,
        top_k=max(candidate_k, top_k),
        candidate_k=max(candidate_k, top_k),
    )
    return reranker.rerank(query, candidates, top_k=top_k)


def load_financial_tables(paths: CompanyPaths) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load annual KPI and growth tables when available."""
    kpis = pd.read_csv(paths.kpis) if paths.kpis.exists() else pd.DataFrame()
    growth = pd.read_csv(paths.growth) if paths.growth.exists() else pd.DataFrame()
    return kpis, growth


def load_evaluation_tables(paths: CompanyPaths) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load retrieval benchmark summary and detailed question-level results."""
    summary = (
        pd.read_csv(paths.evaluation_summary)
        if paths.evaluation_summary.exists()
        else pd.DataFrame()
    )
    details = (
        pd.read_csv(paths.evaluation_details)
        if paths.evaluation_details.exists()
        else pd.DataFrame()
    )
    return summary, details
