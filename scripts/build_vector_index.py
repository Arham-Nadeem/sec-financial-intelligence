#Build a reusable local FAISS semantic index for parsed SEC filing chunks.
from __future__ import annotations

import argparse
from pathlib import Path

from sec_intelligence.retrieval.vector import DEFAULT_MODEL, save_vector_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a local semantic index for SEC filing chunks.")
    parser.add_argument("ticker", help="Stock ticker whose parsed chunks should be indexed.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Sentence-transformers model name.")
    args = parser.parse_args()

    ticker = args.ticker.upper()
    root = Path(__file__).resolve().parents[1]
    chunks_path = root / "data" / "processed" / ticker / "chunks.jsonl"
    index_dir = root / "data" / "indexes" / ticker / "vector"

    metadata = save_vector_index(chunks_path, index_dir, model_name=args.model)

    print(f"Ticker: {ticker}")
    print(f"Embedding model: {metadata['model_name']}")
    print(f"Chunks indexed: {metadata['chunk_count']}")
    print(f"Embedding dimensions: {metadata['embedding_dimension']}")
    print(f"FAISS index: {index_dir / 'chunks.faiss'}")
    print(f"Metadata: {index_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
