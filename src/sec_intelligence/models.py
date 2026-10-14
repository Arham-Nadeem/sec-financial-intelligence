#Shared typed models for filings, chunks, retrieval results, and financial facts.
from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import BaseModel, Field


class FilingMetadata(BaseModel):
    ticker: str
    cik: str
    company_name: str
    form: str
    accession_number: str
    filing_date: date
    report_date: date | None = None
    primary_document: str
    filing_url: str


class FilingSection(BaseModel):
    section_id: str
    title: str
    text: str


class Chunk(BaseModel):
    chunk_id: str
    ticker: str
    form: str
    filing_date: date
    accession_number: str
    section_id: str
    section_title: str
    text: str
    source_url: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    chunk: Chunk
    score: float
    vector_score: float = 0.0
    bm25_score: float = 0.0
    rerank_score: float | None = None


class FinancialPoint(BaseModel):
    metric: str
    fiscal_year: int | None
    fiscal_period: str | None
    form: str
    filed: date
    start: date | None = None
    end: date
    value: float
    unit: str
    accession_number: str | None = None
