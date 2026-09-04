"""Swappable embedding client with local and remote backends.

Implements spec §63 (lightweight, local, fast, stable, reproducible) and
spec §64 (FAISS dense similarity).

Backends (tried in order):
  1. OpenAI-compatible /embeddings HTTP endpoint (if config specifies model + endpoint).
  2. sentence-transformers local model ("all-MiniLM-L6-v2").
  3. TF-IDF fallback via scikit-learn (dimensionality determined by vocabulary).

All returned vectors are L2-normalized (float32, shape (N, D)).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load project config.json, resolving relative to this file's parent."""
    if config_path is not None:
        p = Path(config_path)
    else:
        p = Path(__file__).resolve().parent.parent / "config.json"
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# OpenAI-compatible HTTP backend
# ---------------------------------------------------------------------------


def _embed_http(texts: List[str], endpoint: str, model: str,
                api_key: str = "", batch_size: int = 64) -> np.ndarray:
    """Call an OpenAI-compatible /embeddings endpoint."""
    try:
        import requests
    except ImportError:
        raise RuntimeError(
            "The 'requests' package is required for the HTTP embedding backend. "
            "Install it with: pip install requests"
        )

    all_embeddings: List[List[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        payload: Dict[str, Any] = {
            "model": model,
            "input": batch,
        }
        headers: Dict[str, str] = {
            "Content-Type": "application/json",
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        resp = requests.post(endpoint, json=payload, headers=headers, timeout=120)
        resp.raise_for_status()
        data = resp.json()

        # Handle both OpenAI-style and some alternative API response formats
        if "data" in data:
            items = data["data"]
            # Some APIs return embeddings directly in data; others wrap them
            for item in items:
                emb = item.get("embedding")
                if emb is None:
                    raise RuntimeError(
                        f"Unexpected response format from {endpoint}: "
                        f"missing 'embedding' key in data item: {item}"
                    )
                all_embeddings.append(emb)
        else:
            raise RuntimeError(
                f"Unexpected response format from {endpoint}: "
                f"expected 'data' key in response: {json.dumps(data)[:500]}"
            )

    arr = np.array(all_embeddings, dtype=np.float32)
    return _l2_normalize(arr)


# ---------------------------------------------------------------------------
# sentence-transformers backend
# ---------------------------------------------------------------------------


def _embed_st(texts: List[str], model_name: str = "all-MiniLM-L6-v2",
              batch_size: int = 64) -> np.ndarray:
    """Embed using a local sentence-transformers model."""
    try:
        from sentence_transformers import SentenceTransformer as _ST
    except ImportError:
        raise RuntimeError(
            "sentence-transformers is not installed. "
            "Install it with: pip install sentence-transformers"
        )

    # Lazy load to avoid cold-start cost on every import
    if not hasattr(_embed_st, "_model"):
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
        _embed_st._model = _ST(model_name, device=device)

    model: Any = _embed_st._model
    all_embeddings: List[np.ndarray] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        emb = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        all_embeddings.append(emb)

    if not all_embeddings:
        return np.empty((0, 0), dtype=np.float32)

    arr = np.vstack(all_embeddings).astype(np.float32)
    return _l2_normalize(arr)


# ---------------------------------------------------------------------------
# TF-IDF fallback backend
# ---------------------------------------------------------------------------


def _embed_tfidf(texts: List[str], batch_size: int = 64) -> np.ndarray:
    """TF-IDF fallback using scikit-learn.

    Creates a persistent TfidfVectorizer on first call so that the same
    vocabulary is used across all embed() invocations within this process.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer as _TV
    except ImportError:
        raise RuntimeError(
            "scikit-learn is not installed. "
            "Install it with: pip install scikit-learn"
        )

    if not hasattr(_embed_tfidf, "_vectorizer"):
        _embed_tfidf._vectorizer = _TV(
            sublinear_tf=True,
            dtype=np.float32,
        )
        # Fit on all texts seen so far (lazy — fits on first real call)
        _embed_tfidf._fitted = False

    vec: Any = _embed_tfidf._vectorizer

    if not _embed_tfidf._fitted:
        # Fit on a representative sample to build vocabulary
        sample = texts[:min(500, len(texts))]
        vec.fit(sample)
        _embed_tfidf._fitted = True

    all_rows: List[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        # Re-fit on the batch if needed to capture any new terms
        if not _embed_tfidf._fitted:
            vec.fit(batch)
            _embed_tfidf._fitted = True
        dense = vec.transform(batch).toarray().astype(np.float32)
        all_rows.append(dense)

    if not all_rows:
        return np.empty((0, 0), dtype=np.float32)

    arr = np.vstack(all_rows)
    return _l2_normalize(arr)


# ---------------------------------------------------------------------------
# Normalization utility
# ---------------------------------------------------------------------------


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize each row in-place and return."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # Avoid division by zero for zero vectors
    norms = np.maximum(norms, 1e-12)
    return (vectors / norms).astype(np.float32)


# ---------------------------------------------------------------------------
# Main EmbeddingClient
# ---------------------------------------------------------------------------


class EmbeddingClient:
    """Swappable embedding client.

    Parameters
    ----------
    config : dict or None
        Project config dict (from config.json). If None, loads config.json
        from the project root automatically.
    batch_size : int
        Number of texts to send per API call / batch. Default 64.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 batch_size: int = 64) -> None:
        if config is None:
            config = _load_config()

        self._config = config
        self._batch_size = batch_size
        self._dimensions: Optional[int] = None

        # Determine backend
        emb_cfg = config.get("embeddings", {})
        self._endpoint: Optional[str] = emb_cfg.get("endpoint")
        self._model: Optional[str] = emb_cfg.get("model")

        self._backend = self._select_backend()
        logger.info("EmbeddingClient initialized with backend: %s", self._backend)

    def _select_backend(self) -> str:
        """Choose the best available embedding backend.

        Priority:
          1. HTTP endpoint (if model + endpoint configured)
          2. sentence-transformers (local)
          3. TF-IDF fallback
        """
        # Check for configured HTTP endpoint
        if self._endpoint and self._model and self._model != "TBD":
            logger.info("Using HTTP embedding backend: %s (%s)", self._endpoint, self._model)
            return "http"

        # Try sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer  # noqa: F401
            logger.info("Using sentence-transformers backend: all-MiniLM-L6-v2")
            return "st"
        except ImportError:
            pass

        # Try TF-IDF
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: F401
            logger.info("Using TF-IDF fallback backend")
            return "tfidf"
        except ImportError:
            pass

        raise RuntimeError(
            "No embedding backend available. Configure an endpoint in config.json "
            "(embeddings.model + embeddings.endpoint) or install a local model: "
            "pip install sentence-transformers  (or scikit-learn for TF-IDF fallback)"
        )

    def embed(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts, returning (N, D) float32 L2-normalized array.

        Parameters
        ----------
        texts : list[str]
            Texts to embed. May be empty (returns empty array).

        Returns
        -------
        np.ndarray
            Shape (N, D) where N = len(texts), D = embedding dimensions.

        Raises
        ------
        ValueError
            If texts is not a list or contains non-string elements.
        RuntimeError
            If the selected backend fails.
        """
        if not isinstance(texts, list):
            raise ValueError("texts must be a list of strings")
        for i, t in enumerate(texts):
            if not isinstance(t, str):
                raise ValueError(f"texts[{i}] is not a string: {type(t).__name__}")

        if len(texts) == 0:
            return np.empty((0, 0), dtype=np.float32)

        if self._backend == "http":
            api_key = self._config.get("llm", {}).get("api_key", "")
            result = _embed_http(texts, self._endpoint, self._model,
                                 api_key=api_key, batch_size=self._batch_size)
        elif self._backend == "st":
            result = _embed_st(texts, batch_size=self._batch_size)
        elif self._backend == "tfidf":
            result = _embed_tfidf(texts, batch_size=self._batch_size)
        else:
            raise RuntimeError(f"Unknown backend: {self._backend}")

        # Cache dimensionality from first call
        if self._dimensions is None and result.shape[1] > 0:
            self._dimensions = result.shape[1]

        return result

    @property
    def dimensions(self) -> Optional[int]:
        """Embedding dimensionality, determined after first embed() call."""
        return self._dimensions

    @property
    def backend(self) -> str:
        """Active backend name."""
        return self._backend
