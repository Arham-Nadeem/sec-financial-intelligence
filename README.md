# SEC Financial Intelligence Platform

A financial research system that combines structured SEC XBRL analytics with section-aware retrieval over 10-K and 10-Q filings.

The platform is designed to answer questions such as:

- How has management's discussion of supply-chain risk changed across recent filings?
- What were the largest year-over-year changes in revenue, operating income, and free cash flow?
- Where does management discuss artificial intelligence, and how has that language changed?
- What factors does management identify as drivers of operating performance or liquidity?

Unlike a generic PDF chatbot, the project separates **financial computation**, **document retrieval**, **ranking**, **source grounding**, and **evaluation** into testable components.

## Why This Project Exists

Financial filings mix two different data problems:

1. **Structured analysis** — financial facts, periods, units, and KPI calculations.
2. **Unstructured research** — management commentary, risk disclosures, competitive language, and changes in narrative over time.

A useful research system needs both. This project uses SEC Company Facts for standardized financial metrics while independently processing filing text for source-grounded research.

## Architecture

```text
SEC EDGAR
├── Company submissions API
├── Filing HTML documents
└── Company Facts XBRL API
        │
        ▼
Ingestion + Local Cache
        │
        ├───────────────────────────────┐
        ▼                               ▼
Filing Text Pipeline                Financial Pipeline
HTML cleanup                        XBRL concept mapping
Section detection                   Period normalization
Section-aware chunking              Annual KPI calculations
Metadata preservation               YoY trend preparation
        │                               │
        ▼                               ▼
Local Search Index                 Structured KPI tables
BM25 + embeddings                         │
        │                                  │
        ├───────────────┐                  │
        ▼               ▼                  │
Hybrid ranking      Cross-encoder           │
                    reranking               │
        └───────────────┬───────────────────┘
                        ▼
              Research Interface
          Ranked evidence + citations
              Financial trend view
                        │
                        ▼
                    Evaluation
        Recall@K · MRR · nDCG · keyword hits
        BM25 vs vector vs hybrid vs reranked
```

## Core Technical Features

### SEC filing ingestion

- Resolves ticker symbols to SEC CIK identifiers.
- Retrieves recent 10-K and 10-Q metadata from SEC submissions data.
- Downloads primary filing HTML from the SEC archives.
- Uses a declared SEC user agent and configurable request throttling.
- Stores downloaded source documents locally so repeated analysis does not repeatedly hit SEC endpoints.

### Section-aware filing parsing

Instead of splitting an entire filing into arbitrary windows, the parser first identifies filing sections such as:

- Item 1 — Business
- Item 1A — Risk Factors
- Item 7 — Management's Discussion and Analysis
- 10-Q Management's Discussion and Analysis
- 10-Q Risk Factors

Chunks retain:

- ticker
- filing type
- filing date
- accession number
- filing section
- source URL
- stable chunk ID

This metadata makes retrieval results explainable and citation-ready.

### Hybrid retrieval

The search layer combines:

- **BM25 lexical retrieval** for exact financial and filing terminology
- **Sentence-transformer embeddings** for semantic similarity
- **FAISS** for local vector search
- **Weighted hybrid scoring** to combine lexical and semantic evidence
- **Optional local cross-encoder reranking** for stronger final ranking

The default models run locally after downloading their model weights.

### Structured financial analytics

The platform reads SEC Company Facts and maps common US-GAAP concepts into normalized metrics including:

- revenue
- operating income
- net income
- operating cash flow
- capital expenditures
- free cash flow

The analytics layer prepares annual observations and year-over-year comparisons without relying on an LLM to perform arithmetic.

### Source-grounded research answers

The default answer layer is intentionally extractive and citation-first. Retrieved evidence is presented with:

- filing type
- filing date
- filing section
- chunk identifier
- direct SEC source URL

An LLM can be added later as an optional synthesis layer, but the retrieval and financial-analysis pipeline does not depend on one.

## Retrieval Evaluation

The repository includes a starter benchmark of filing-research questions and evaluates multiple strategies on the same question set.

Metrics include:

- **Recall@K** — whether relevant filing sections were retrieved
- **MRR** — how early the first relevant result appears
- **nDCG@K** — ranking quality across the returned evidence
- **Keyword Evidence Hit Rate** — whether expected evidence appears in the retrieved context

The evaluation script compares:

1. BM25 only
2. Vector retrieval only
3. Hybrid retrieval
4. Hybrid retrieval + cross-encoder reranking

This makes retrieval choices measurable rather than subjective.

## Project Structure

```text
sec-financial-intelligence/
├── config/
│   └── settings.yaml
├── data/
│   ├── evaluation/
│   │   └── questions.yaml
│   ├── indexes/
│   ├── processed/
│   └── raw/
├── scripts/
│   ├── build_company.py
│   ├── evaluate_retrieval.py
│   └── run_app.py
├── src/sec_intelligence/
│   ├── analytics/
│   │   └── financials.py
│   ├── answering/
│   │   └── grounded.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── runner.py
│   ├── ingestion/
│   │   ├── cache.py
│   │   └── sec_client.py
│   ├── parsing/
│   │   ├── chunker.py
│   │   └── filing_parser.py
│   ├── retrieval/
│   │   └── hybrid.py
│   ├── ui/
│   │   └── app.py
│   ├── config.py
│   ├── models.py
│   └── pipeline.py
├── tests/
│   ├── test_chunker.py
│   ├── test_financials.py
│   ├── test_metrics.py
│   └── test_parser.py
├── .gitignore
├── pyproject.toml
└── README.md
```

## Setup

Python 3.11+ is recommended.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install the project and development tools:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Configure SEC Access

Edit `config/settings.yaml` and replace the placeholder `user_agent` value with your name and contact email.

Example:

```yaml
sec:
  user_agent: "Your Name your.email@example.com"
```

No SEC API key is required for the public data used by this project.

## Build a Company Dataset

Example using Apple:

```bash
python scripts/build_company.py AAPL
```

This creates local processed artifacts such as:

```text
data/processed/AAPL/chunks.jsonl
data/processed/AAPL/financial_facts.csv
data/processed/AAPL/annual_kpis.csv
data/processed/AAPL/filings.json
```

Raw filing HTML and Company Facts responses are cached under `data/raw/AAPL/`.

## Run the Retrieval Evaluation

```bash
python scripts/evaluate_retrieval.py AAPL --k 5
```

The script writes per-strategy results and a comparison summary to:

```text
data/processed/AAPL/evaluation/
```

The benchmark in `data/evaluation/questions.yaml` is intentionally editable. A strong portfolio version should expand it with manually reviewed, company-specific questions and labels.

## Run the Streamlit Interface

```bash
python scripts/run_app.py
```

Or:

```bash
python -m streamlit run src/sec_intelligence/ui/app.py
```

The initial interface includes:

- Research Assistant
- Financial Trends
- Evidence Explorer

## Testing

```bash
pytest
```

The tests cover section parsing, chunk metadata, structured financial calculations, and evaluation metrics without requiring live SEC requests.

## What Is Fully Free / Local

After SEC documents and model weights are downloaded, the core analytics and search workflow runs locally.

| Component | Implementation | Cost |
|---|---|---:|
| Filing metadata | SEC submissions data | Free |
| Filing documents | SEC EDGAR archives | Free |
| Structured financial facts | SEC Company Facts | Free |
| Parsing | BeautifulSoup + Python | Free/local |
| Lexical retrieval | BM25 | Free/local |
| Embeddings | sentence-transformers | Free/local |
| Vector search | FAISS | Free/local |
| Reranking | local cross-encoder | Free/local |
| Financial analytics | pandas | Free/local |
| UI | Streamlit | Free/local |
| Tests/evaluation | pytest + custom metrics | Free/local |
| LLM synthesis | Optional future layer | Not required |

## Planned Portfolio Enhancements

The strongest next iterations are:

- filing-to-filing narrative comparison for the same section
- richer quarterly KPI normalization
- company-specific retrieval benchmark with manually verified relevance labels
- answer-level citation-support checks
- retrieval latency measurements
- persisted FAISS indexes instead of rebuilding on app startup
- multi-company comparison
- optional local LLM synthesis with strict citation constraints
- screenshots and an architecture diagram for the final GitHub README

## Important Data Note

SEC filings are public source documents. Model outputs and retrieved evidence should still be treated as research assistance rather than investment advice. Financial concepts can vary by issuer, taxonomy choice, and reporting period, so structured metrics should be validated when used for decision-making.
