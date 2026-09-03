"""FAISS index management.

Implements spec §64 (FAISS: simple, local, fast, transparent).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class FAISSIndex:
    """Thin wrapper around FAISS index for paragraph-level retrieval.

    Parameters
    ----------
    dimensions : int
        Embedding vector dimensionality.
    metric : str
        "cosine" for cosine similarity (uses L2-normalized vectors + InnerProduct),
        "l2" for Euclidean distance.
    """

    def __init__(self, dimensions: int, metric: str = "cosine") -> None:
        if dimensions <= 0:
            raise ValueError(f"dimensions must be positive, got {dimensions}")
        if metric not in ("cosine", "l2"):
            raise ValueError(f"metric must be 'cosine' or 'l2', got '{metric}'")

        self._dimensions = dimensions
        self._metric = metric

        # Import FAISS lazily
        import faiss  # noqa: F811

        if metric == "cosine":
            # For cosine similarity with FAISS:
            # L2-normalize vectors, then use InnerProduct index.
            # FAISS IP returns dot product = cosine similarity for unit vectors.
            self._index = faiss.IndexFlatIP(dimensions)
        else:
            self._index = faiss.IndexFlatL2(dimensions)

        self._metadata: list[str] = []  # paragraph id for each vector

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def add(self, vectors: np.ndarray) -> None:
        """Add vectors to the index.

        Parameters
        ----------
        vectors : np.ndarray
            Shape (N, D) float32 array. Must match index dimensions.

        Raises
        ------
        ValueError
            If vector dimensions don't match or input is malformed.
        """
        if vectors.size == 0:
            return  # Edge case: adding nothing is a no-op

        if not isinstance(vectors, np.ndarray):
            raise ValueError("vectors must be a numpy array")

        if vectors.ndim != 2:
            raise ValueError(f"vectors must be 2D (N, D), got {vectors.ndim}D")

        if vectors.shape[1] != self._dimensions:
            raise ValueError(
                f"Vector dimension mismatch: index={self._dimensions}, "
                f"vectors={vectors.shape[1]}"
            )

        vectors = vectors.astype(np.float32, copy=False)

        n_added = vectors.shape[0]
        self._index.add(vectors)
        self._metadata.extend([f"__vec_{i}__" for i in range(n_added)])

    def add_batch(self, vectors: np.ndarray, ids: list[str]) -> None:
        """Add vectors with explicit paragraph IDs (alias for add_with_ids)."""
        self.add_with_ids(vectors, ids)

    def add_with_ids(self, vectors: np.ndarray, ids: list[str]) -> None:
        """Add vectors with explicit paragraph IDs.

        Parameters
        ----------
        vectors : np.ndarray
            Shape (N, D) float32 array.
        ids : list[str]
            Paragraph ID for each vector (must match N).
        """
        if len(ids) != vectors.shape[0]:
            raise ValueError(
                f"ids length ({len(ids)}) must match vectors count ({vectors.shape[0]})"
            )
        self.add(vectors)
        # Replace auto-generated metadata with provided IDs
        start = len(self._metadata) - len(ids)
        self._metadata[start:start + len(ids)] = ids

    def search(self, query_vector: np.ndarray, k: int = 20
               ) -> Tuple[np.ndarray, np.ndarray]:
        """Search the index for the k nearest neighbors.

        Parameters
        ----------
        query_vector : np.ndarray
            Single query vector, shape (D,) or (1, D).
        k : int
            Number of neighbors to retrieve. Clamped to [1, count()].

        Returns
        -------
        distances : np.ndarray
            Shape (1, k) — similarity scores (higher = more similar for cosine).
        indices : np.ndarray
            Shape (1, k) — integer indices into the index. -1 if slot empty.

        Raises
        ------
        RuntimeError
            If index is empty or query dimension mismatch.
        """
        if self.count() == 0:
            raise RuntimeError("Cannot search an empty index")

        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        if query_vector.shape[1] != self._dimensions:
            raise ValueError(
                f"Query dimension mismatch: index={self._dimensions}, "
                f"query={query_vector.shape[1]}"
            )

        query_vector = query_vector.astype(np.float32, copy=False)

        # Clamp k to available vectors
        k = max(1, min(k, self.count()))

        distances, indices = self._index.search(query_vector, k)
        return distances.astype(np.float32), indices.astype(np.int64)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save index and metadata to disk.

        Parameters
        ----------
        path : str
            Directory path. Saves:
              - index.faiss  (FAISS binary)
              - metadata.json (paragraph id → index mapping)
        """
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)

        import faiss  # noqa: F811
        faiss.write_index(self._index, str(p / "index.faiss"))

        meta_path = p / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "dimensions": self._dimensions,
                "metric": self._metric,
                "ids": self._metadata,
            }, f, indent=2)

        logger.info("Saved FAISSIndex (%d vectors, %d dims) to %s",
                     self.count(), self._dimensions, p)

    @classmethod
    def load(cls, path: str) -> FAISSIndex:
        """Load index and metadata from disk.

        Parameters
        ----------
        path : str
            Directory path containing index.faiss and metadata.json.

        Returns
        -------
        FAISSIndex
            Restored index instance.
        """
        p = Path(path)

        meta_path = p / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        dimensions = meta["dimensions"]
        metric = meta["metric"]

        import faiss  # noqa: F811
        index = faiss.read_index(str(p / "index.faiss"))

        instance = cls.__new__(cls)
        instance._dimensions = dimensions
        instance._metric = metric
        instance._index = index
        instance._metadata = meta["ids"]

        logger.info("Loaded FAISSIndex (%d vectors, %d dims) from %s",
                     len(instance._metadata), dimensions, p)
        return instance

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Number of vectors in the index."""
        return self._index.ntotal

    @property
    def metadata(self) -> list[str]:
        """List of paragraph IDs corresponding to each vector."""
        return list(self._metadata)

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def metric(self) -> str:
        return self._metric
