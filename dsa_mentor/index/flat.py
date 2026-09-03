"""Flat paragraph-level retrieval.

Implements spec §87 Phase 2: flat retrieval baseline.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from ..embeddings import EmbeddingClient
from ..models import Paragraph
from .base import FAISSIndex

logger = logging.getLogger(__name__)


class FlatRetriever:
    """Flat paragraph-level retriever using FAISS + embeddings.

    Parameters
    ----------
    embedding_client : EmbeddingClient or None
        If None, creates one from config.json.
    index_path : str or None
        Directory to save/load the FAISS index. If None, index is kept in memory.
    """

    def __init__(self, embedding_client: Optional[EmbeddingClient] = None,
                 index_path: Optional[str] = None) -> None:
        self._embedding_client = embedding_client
        self._index_path = index_path
        self._index: Optional[FAISSIndex] = None
        self._paragraphs: List[Paragraph] = []
        self._id_to_index: dict[str, int] = {}  # paragraph id → vector index

    @property
    def embedding_client(self) -> EmbeddingClient:
        """Lazy-init embedding client from config.json."""
        if self._embedding_client is None:
            from ..embeddings import EmbeddingClient
            self._embedding_client = EmbeddingClient()
        return self._embedding_client

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index(self, paragraphs: List[Paragraph]) -> None:
        """Embed all paragraphs and build the FAISS index.

        Parameters
        ----------
        paragraphs : list[Paragraph]
            Paragraphs to index. Empty list is a no-op.
        """
        if not paragraphs:
            logger.warning("index() called with empty paragraph list — no-op")
            return

        # Extract text content
        texts = []
        valid_ids = []
        for p in paragraphs:
            if p.content is None or p.content.strip() == "":
                logger.warning("Skipping paragraph %s: empty content", p.id)
                continue
            texts.append(p.content)
            valid_ids.append(p.id)

        if not texts:
            logger.warning("All paragraphs had empty content — no indexing")
            return

        # Embed in batches
        logger.info("Embedding %d paragraphs...", len(texts))
        vectors = self.embedding_client.embed(texts)
        logger.info("Embedding complete: shape=%s, backend=%s",
                     vectors.shape, self.embedding_client.backend)

        # Build index
        dims = vectors.shape[1]
        self._index = FAISSIndex(dims, metric="cosine")
        self._index.add_with_ids(vectors, valid_ids)

        self._paragraphs = paragraphs
        self._id_to_index = {p.id: i for i, p in enumerate(paragraphs)
                             if p.content is not None and p.content.strip() != ""}

        # Save to disk if path specified
        if self._index_path is not None:
            self.save()

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, k: int = 20
               ) -> List[Tuple[Paragraph, float]]:
        """Search for the k most similar paragraphs.

        Parameters
        ----------
        query : str
            Query text to embed and search.
        k : int
            Number of results. Clamped to [1, index count].

        Returns
        -------
        list[tuple[Paragraph, float]]
            Sorted by descending similarity. Each tuple is (paragraph, similarity).
            Similarity = dot product (cosine similarity for L2-normalized vectors).

        Raises
        ------
        RuntimeError
            If the index has not been built or loaded.
        """
        if self._index is None:
            raise RuntimeError("Index not built. Call index() or load() first.")

        query_vec = self.embedding_client.embed([query])
        distances, indices = self._index.search(query_vec, k=k)

        results: List[Tuple[Paragraph, float]] = []
        for j in range(distances.shape[1]):
            idx = int(indices[0, j])
            dist = float(distances[0, j])
            if idx == -1:
                continue
            # Map FAISS index back to paragraph
            para_id = self._index.metadata[idx]
            # Find the paragraph in our list
            para = self._get_paragraph_by_id(para_id)
            if para is not None:
                results.append((para, dist))

        # Sort by descending similarity (FAISS IP already returns descending,
        # but be explicit for safety)
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Save index + metadata to disk."""
        if self._index is None:
            raise RuntimeError("Cannot save: index not built")
        if self._index_path is None:
            raise RuntimeError("Cannot save: no index_path configured")

        self._index.save(self._index_path)

        # Save paragraph list and id→index mapping as JSON
        meta = {
            "paragraph_ids": [p.id for p in self._paragraphs],
            "id_to_index": self._id_to_index,
        }
        meta_path = Path(self._index_path) / "retriever_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        logger.info("Saved FlatRetriever metadata to %s", meta_path)

    def load(self) -> None:
        """Load index + metadata from disk."""
        if self._index_path is None:
            raise RuntimeError("Cannot load: no index_path configured")

        self._index = FAISSIndex.load(self._index_path)

        # Load paragraph metadata
        meta_path = Path(self._index_path) / "retriever_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Retriever metadata not found at {meta_path}. "
                f"Call index() first to build and save."
            )

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        # Reconstruct paragraphs from saved IDs (paragraphs must be provided
        # separately if full objects aren't available)
        self._id_to_index = meta.get("id_to_index", {})
        self._paragraphs = []  # Will be populated when paragraphs are provided

    def load_paragraphs(self, paragraphs: List[Paragraph]) -> None:
        """Associate loaded index with paragraph objects.

        Call this after load() to restore paragraph lookup.
        """
        self._paragraphs = paragraphs
        self._id_to_index = {p.id: i for i, p in enumerate(paragraphs)
                             if p.content is not None and p.content.strip() != ""}

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Number of indexed paragraphs."""
        if self._index is None:
            return 0
        return self._index.count()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_paragraph_by_id(self, para_id: str) -> Optional[Paragraph]:
        """Look up a paragraph by its ID."""
        for p in self._paragraphs:
            if p.id == para_id:
                return p
        return None


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def flat_search(paragraphs: List[Paragraph], query: str, k: int = 20,
                index_path: Optional[str] = None,
                embedding_client: Optional[EmbeddingClient] = None
                ) -> List[Tuple[Paragraph, float]]:
    """Convenience function for one-off flat searches.

    Builds (or loads) an index, searches, and returns results.
    The index is cached on disk for subsequent calls with the same index_path.

    Parameters
    ----------
    paragraphs : list[Paragraph]
        Paragraphs to search.
    query : str
        Query text.
    k : int
        Number of results.
    index_path : str or None
        Path for caching the index. If provided and index exists, loads it.
    embedding_client : EmbeddingClient or None
        Custom embedding client. If None, creates one from config.

    Returns
    -------
    list[tuple[Paragraph, float]]
        (paragraph, similarity) sorted by descending similarity.
    """
    retriever = FlatRetriever(
        embedding_client=embedding_client,
        index_path=index_path,
    )

    if index_path is not None:
        # Try to load existing index
        meta_path = Path(index_path) / "index.faiss"
        if meta_path.exists():
            try:
                retriever.load()
                retriever.load_paragraphs(paragraphs)
                return retriever.search(query, k=k)
            except Exception as e:
                logger.warning("Failed to load index from %s: %s — rebuilding",
                               index_path, e)

    # Build fresh index
    retriever.index(paragraphs)
    return retriever.search(query, k=k)
