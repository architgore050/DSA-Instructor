"""FAISS index management.

Implements spec §64 (FAISS: simple, local, fast, transparent).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional, Tuple

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

        # Auto-detect GPU: use GPU index if CUDA is available and faiss has
        # GPU support; otherwise fall back to CPU.  No config flag needed —
        # the code just works when CUDA + faiss-gpu are installed.
        self._use_gpu = self._detect_gpu()
        self._resources: Any | None = None

        if self._use_gpu:
            self._resources = faiss.StandardGpuResources()
            if metric == "cosine":
                self._index = faiss.GpuIndexFlatIP(
                    self._resources, dimensions, faiss.FLOAT32
                )
            else:
                self._index = faiss.GpuIndexFlatL2(
                    self._resources, dimensions, faiss.FLOAT32
                )
            logger.info("Using GPU FAISS index (%d dims, %s)", dimensions, metric)
        else:
            if metric == "cosine":
                self._index = faiss.IndexFlatIP(dimensions)
            else:
                self._index = faiss.IndexFlatL2(dimensions)
            logger.info("Using CPU FAISS index (%d dims, %s)", dimensions, metric)

        self._metadata: list[str] = []  # paragraph id for each vector

    @staticmethod
    def _detect_gpu() -> bool:
        """Return True if CUDA is available and FAISS has GPU support."""
        try:
            import torch  # noqa: F811
            if not torch.cuda.is_available():
                return False
        except ImportError:
            pass
        # Verify FAISS can actually create a GPU index (faiss-gpu installed)
        try:
            import faiss  # noqa: F811
            res = faiss.StandardGpuResources()
            idx = faiss.GpuIndexFlatIP(res, 8, faiss.FLOAT32)
            del idx, res
            return True
        except Exception:
            return False

    def _to_cpu_index(self) -> Any:
        """Copy a GPU index to CPU for serialization."""
        import faiss  # noqa: F811
        if self._use_gpu:
            return faiss.index_gpu_to_cpu(self._index)
        return self._index

    def _to_gpu_index(self, cpu_index: Any) -> Any:
        """Copy a CPU index to GPU."""
        import faiss  # noqa: F811
        if self._use_gpu and self._resources is not None:
            return faiss.index_cpu_to_gpu(self._resources, 0, cpu_index)
        return cpu_index

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
              - index.faiss  (FAISS binary, always stored as CPU index)
              - metadata.json (paragraph id → index mapping)
        """
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)

        import faiss  # noqa: F811
        # Always serialize as CPU index — GPU indices cannot be written
        # directly; copy to CPU first.
        faiss.write_index(self._to_cpu_index(), str(p / "index.faiss"))

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
            Restored index instance (re-creates GPU index if CUDA is available).
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
        cpu_index = faiss.read_index(str(p / "index.faiss"))

        instance = cls.__new__(cls)
        instance._dimensions = dimensions
        instance._metric = metric
        instance._use_gpu = instance._detect_gpu()
        instance._resources: Any | None = None

        if instance._use_gpu:
            instance._resources = faiss.StandardGpuResources()
            if metric == "cosine":
                instance._index = faiss.GpuIndexFlatIP(
                    instance._resources, dimensions, faiss.FLOAT32
                )
            else:
                instance._index = faiss.GpuIndexFlatL2(
                    instance._resources, dimensions, faiss.FLOAT32
                )
            # Copy loaded CPU data onto GPU
            instance._index.copyFrom(cpu_index)
            logger.info(
                "Loaded FAISSIndex (%d vectors, %d dims) from %s [GPU]",
                len(meta["ids"]), dimensions, p,
            )
        else:
            instance._index = cpu_index
            logger.info(
                "Loaded FAISSIndex (%d vectors, %d dims) from %s [CPU]",
                len(meta["ids"]), dimensions, p,
            )

        instance._metadata = meta["ids"]
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
