#Build and search local semantic indexes for SEC filing chunks with sentence embeddings and FAISS.
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Protocol

import numpy as np

from sec_intelligence.models import Chunk, RetrievalResult
from sec_intelligence.retrieval.bm25 import load_chunks


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class TextEncoder(Protocol):
    """Minimal encoder interface used by the vector retriever and tests."""

    def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
        ...


class SentenceTransformerEncoder:
    """Lazy wrapper around a local sentence-transformers embedding model."""

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
        return np.asarray(
            self.model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=bool(kwargs.get("show_progress_bar", False)),
                batch_size=int(kwargs.get("batch_size", 32)),
            ),
            dtype=np.float32,
        )


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize embeddings so inner product equals cosine similarity."""
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("Embeddings must be a two-dimensional matrix.")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def chunk_file_hash(path: Path) -> str:
    """Hash the parsed chunk file so saved indexes can detect stale source data."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class VectorRetriever:
    """Rank filing chunks by semantic cosine similarity."""

    def __init__(
        self,
        chunks: list[Chunk],
        encoder: TextEncoder,
        embeddings: np.ndarray | None = None,
    ) -> None:
        if not chunks:
            raise ValueError("At least one chunk is required to build a vector index.")

        self.chunks = chunks
        self.encoder = encoder
        if embeddings is None:
            embeddings = encoder.encode(
                [chunk.text for chunk in chunks],
                show_progress_bar=True,
                batch_size=32,
            )
        self.embeddings = _normalize_rows(embeddings)
        if len(self.embeddings) != len(chunks):
            raise ValueError("Embedding count must match chunk count.")

    def search(
        self,
        query: str,
        top_k: int = 5,
        form: str | None = None,
        section_id: str | None = None,
    ) -> list[RetrievalResult]:
        """Return semantically similar chunks with optional filing/section filters."""
        if not query.strip():
            raise ValueError("Query cannot be empty.")
        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        #Embeddings are normalized, so the dot product is cosine similarity.
        query_embedding = _normalize_rows(self.encoder.encode([query]))[0]
        similarities = self.embeddings @ query_embedding
        normalized_form = form.upper() if form else None
        normalized_section = section_id.lower() if section_id else None

        candidates: list[tuple[int, float]] = []
        for index, score in enumerate(similarities):
            chunk = self.chunks[index]
            if normalized_form and chunk.form.upper() != normalized_form:
                continue
            if normalized_section and chunk.section_id.lower() != normalized_section:
                continue
            candidates.append((index, float(score)))

        candidates.sort(key=lambda row: row[1], reverse=True)
        return [
            RetrievalResult(
                chunk=self.chunks[index],
                score=score,
                vector_score=score,
            )
            for index, score in candidates[:top_k]
        ]


def build_faiss_index(embeddings: np.ndarray):
    """Create a FAISS inner-product index over normalized embeddings."""
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError("FAISS is not installed. Run: python -m pip install -e .") from exc

    normalized = _normalize_rows(embeddings)
    index = faiss.IndexFlatIP(normalized.shape[1])
    index.add(normalized)
    return index


def save_vector_index(
    chunks_path: Path,
    output_dir: Path,
    model_name: str = DEFAULT_MODEL,
    encoder: TextEncoder | None = None,
) -> dict[str, object]:
    """Encode chunks and persist a reusable FAISS index with metadata."""
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError("FAISS is not installed. Run: python -m pip install -e .") from exc

    chunks = load_chunks(chunks_path)
    active_encoder = encoder or SentenceTransformerEncoder(model_name)
    embeddings = _normalize_rows(
        active_encoder.encode(
            [chunk.text for chunk in chunks],
            show_progress_bar=True,
            batch_size=32,
        )
    )
    index = build_faiss_index(embeddings)

    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "chunks.faiss"
    metadata_path = output_dir / "metadata.json"
    faiss.write_index(index, str(index_path))

    #Persist enough metadata to detect when parsed chunks change and the index becomes stale.
    metadata = {
        "model_name": model_name,
        "chunk_count": len(chunks),
        "embedding_dimension": int(embeddings.shape[1]),
        "chunks_sha256": chunk_file_hash(chunks_path),
        "chunk_ids": [chunk.chunk_id for chunk in chunks],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def load_saved_vector_retriever(
    chunks_path: Path,
    index_dir: Path,
    encoder: TextEncoder | None = None,
) -> VectorRetriever:
    """Load persisted FAISS vectors and verify they match the current chunks."""
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError("FAISS is not installed. Run: python -m pip install -e .") from exc

    metadata_path = index_dir / "metadata.json"
    index_path = index_dir / "chunks.faiss"
    if not metadata_path.exists() or not index_path.exists():
        raise FileNotFoundError("Vector index not found. Build it with scripts/build_vector_index.py first.")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("chunks_sha256") != chunk_file_hash(chunks_path):
        raise ValueError("Vector index is stale because chunks.jsonl changed. Rebuild the vector index.")

    chunks = load_chunks(chunks_path)
    if metadata.get("chunk_count") != len(chunks):
        raise ValueError("Vector index chunk count does not match chunks.jsonl. Rebuild the vector index.")

    index = faiss.read_index(str(index_path))
    embeddings = np.vstack([index.reconstruct(i) for i in range(index.ntotal)]).astype(np.float32)
    active_encoder = encoder or SentenceTransformerEncoder(str(metadata["model_name"]))
    return VectorRetriever(chunks=chunks, encoder=active_encoder, embeddings=embeddings)
