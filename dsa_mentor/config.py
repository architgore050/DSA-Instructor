"""Centralized configuration for DSA Mentor (spec §34).

Every tunable of the project lives in a single ``config.json`` at the workspace
root. This module loads that file, validates it with clear error messages, and
exposes it as typed, attribute-accessible objects:

    from dsa_mentor.config import load_config

    cfg = load_config()                      # or load_config(path) for tests
    model_id = cfg.llm.models[cfg.llm.default_model]
    k = cfg.retrieval.book_knee.candidate_k  # -> 10
    threshold = cfg.retrieval.similarity_threshold

The default config path is resolved relative to this file (``<root>/dsa_mentor/
config.py`` → ``<root>/config.json``), so loading works from any CWD. Unknown
extra keys are preserved in :attr:`Config.raw` for forward compatibility with
later phases (e.g. the spec's ``evaluation`` section).

This module is dependency-free (stdlib only).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

CONFIG_FILENAME = "config.json"


class ConfigError(Exception):
    """Raised when config.json is missing, unreadable, malformed, or invalid."""


# ---------------------------------------------------------------------------
# Field extraction helpers (each raises ConfigError with a full dotted path)
# ---------------------------------------------------------------------------

def _path(where: str, key: str) -> str:
    """Dotted config path for error messages ('' + 'llm' → 'llm')."""
    return f"{where}.{key}" if where else key


def _fail(where: str, problem: str) -> "ConfigError":
    return ConfigError(f"invalid config key '{where}': {problem}")


def _require(mapping: Any, key: str, where: str) -> Any:
    if not isinstance(mapping, dict):
        raise _fail(where or "<root>", f"expected an object, got {_type_name(mapping)}")
    if key not in mapping:
        raise ConfigError(f"missing required config key '{_path(where, key)}' (in {CONFIG_FILENAME})")
    return mapping[key]


def _get_str(data: dict, key: str, where: str) -> str:
    value = _require(data, key, where)
    if not isinstance(value, str):
        raise _fail(_path(where, key), f"expected a string, got {_type_name(value)}")
    return value


def _get_optional_str(data: dict, key: str, where: str) -> Optional[str]:
    value = _require(data, key, where)
    if value is None or isinstance(value, str):
        return value
    raise _fail(_path(where, key), f"expected a string or null, got {_type_name(value)}")


def _get_bool(data: dict, key: str, where: str) -> bool:
    value = _require(data, key, where)
    if not isinstance(value, bool):
        raise _fail(_path(where, key), f"expected a boolean, got {_type_name(value)}")
    return value


def _get_optional_int(data: dict, key: str, where: str) -> Optional[int]:
    value = _require(data, key, where)
    if value is None or (isinstance(value, int) and not isinstance(value, bool)):
        return value
    raise _fail(_path(where, key), f"expected an integer or null, got {_type_name(value)}")


def _get_int(data: dict, key: str, where: str) -> int:
    value = _require(data, key, where)
    if isinstance(value, bool) or not isinstance(value, int):
        raise _fail(_path(where, key), f"expected an integer, got {_type_name(value)}")
    return value


def _get_number(data: dict, key: str, where: str) -> float:
    value = _require(data, key, where)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _fail(_path(where, key), f"expected a number, got {_type_name(value)}")
    return float(value)


def _get_dict(data: dict, key: str, where: str) -> dict:
    value = _require(data, key, where)
    if not isinstance(value, dict):
        raise _fail(_path(where, key), f"expected an object, got {_type_name(value)}")
    return value


def _type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


# ---------------------------------------------------------------------------
# Typed sections
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SamplingConfig:
    """Default sampling parameters applied to every chat request."""

    temperature: float
    top_p: float
    top_k: int
    max_tokens: int

    @classmethod
    def from_dict(cls, data: dict) -> "SamplingConfig":
        where = "llm.sampling"
        return cls(
            temperature=_get_number(data, "temperature", where),
            top_p=_get_number(data, "top_p", where),
            top_k=_get_int(data, "top_k", where),
            max_tokens=_get_int(data, "max_tokens", where),
        )


@dataclass(frozen=True)
class LLMConfig:
    """LLM endpoint + model tiers (see EXECUTION_NOTES.md)."""

    base_url: str
    api_key: str
    models: dict  # tier name ("large"/"medium"/"small") -> model id
    default_model: str  # a tier name, or an explicit model id
    timeout_seconds: float
    max_retries: int
    retry_backoff_base_seconds: float
    retry_max_delay_seconds: float
    sampling: SamplingConfig

    @classmethod
    def from_dict(cls, data: dict) -> "LLMConfig":
        where = "llm"
        models_raw = _get_dict(data, "models", where)
        if not models_raw:
            raise _fail(f"{where}.models", "must be a non-empty object of tier name -> model id")
        models = {}
        for tier, model_id in models_raw.items():
            if not isinstance(tier, str) or not tier.strip():
                raise _fail(f"{where}.models", f"tier names must be non-empty strings (got {tier!r})")
            if not isinstance(model_id, str) or not model_id.strip():
                raise _fail(
                    f"{where}.models.{tier}",
                    f"expected a non-empty model id string, got {_type_name(model_id)}",
                )
            models[tier] = model_id
        return cls(
            base_url=_get_str(data, "base_url", where),
            api_key=_get_str(data, "api_key", where),  # empty string is a valid placeholder
            models=models,
            default_model=_get_str(data, "default_model", where),
            timeout_seconds=_get_number(data, "timeout_seconds", where),
            max_retries=_get_int(data, "max_retries", where),
            retry_backoff_base_seconds=_get_number(data, "retry_backoff_base_seconds", where),
            retry_max_delay_seconds=_get_number(data, "retry_max_delay_seconds", where),
            sampling=SamplingConfig.from_dict(_get_dict(data, "sampling", where)),
        )


@dataclass(frozen=True)
class ExperimentConfig:
    """Master experiment switches (spec §34)."""

    rag_enabled: bool  # master switch; when OFF there is no retrieval tool either (§71–§72)

    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentConfig":
        return cls(rag_enabled=_get_bool(data, "rag_enabled", "experiment"))


@dataclass(frozen=True)
class KneeParams:
    """Per-index knee-detection bounds (spec §12–§13)."""

    candidate_k: int  # size of the similarity-sorted candidate pool
    minimum: int  # always retain at least this many
    maximum: int  # never retain more than this many


def _knee_params(data: dict, key: str, where: str) -> KneeParams:
    raw = _get_dict(data, key, where)
    return KneeParams(
        candidate_k=_get_int(raw, "candidate_k", f"{where}.{key}"),
        minimum=_get_int(raw, "minimum", f"{where}.{key}"),
        maximum=_get_int(raw, "maximum", f"{where}.{key}"),
    )


@dataclass(frozen=True)
class RetrievalConfig:
    """Retrieval + context-building tunables (spec §6, §9–§14, §18)."""

    knee_method: str  # algorithm name placeholder; one universal method at all levels (§12)
    similarity_threshold: float  # fallback relevance floor when no strong knee exists (§13)
    neighbor_window: int  # adjacent-paragraph expansion window (§17)
    max_context_tokens: int  # context budget for the LLM prompt (§18)
    paragraph_max_chars: int  # oversized-paragraph split threshold P (spec §6)
    paragraph_overlap_chars: int  # overlap between split subparagraph chunks (spec §6)
    book_knee: KneeParams
    chapter_knee: KneeParams
    topic_knee: KneeParams
    paragraph_knee: KneeParams

    @classmethod
    def from_dict(cls, data: dict) -> "RetrievalConfig":
        where = "retrieval"
        return cls(
            knee_method=_get_str(data, "knee_method", where),
            similarity_threshold=_get_number(data, "similarity_threshold", where),
            neighbor_window=_get_int(data, "neighbor_window", where),
            max_context_tokens=_get_int(data, "max_context_tokens", where),
            paragraph_max_chars=_get_int(data, "paragraph_max_chars", where),
            paragraph_overlap_chars=_get_int(data, "paragraph_overlap_chars", where),
            book_knee=_knee_params(data, "book_knee", where),
            chapter_knee=_knee_params(data, "chapter_knee", where),
            topic_knee=_knee_params(data, "topic_knee", where),
            paragraph_knee=_knee_params(data, "paragraph_knee", where),
        )


@dataclass(frozen=True)
class AgenticRetrievalConfig:
    """Agentic (tool-based) retrieval switches (spec §23–§25)."""

    enabled: bool
    max_tool_calls: int  # hard cap on tool invocations per question (§25)

    @classmethod
    def from_dict(cls, data: dict) -> "AgenticRetrievalConfig":
        where = "agentic_retrieval"
        return cls(
            enabled=_get_bool(data, "enabled", where),
            max_tool_calls=_get_int(data, "max_tool_calls", where),
        )


@dataclass(frozen=True)
class EmbeddingsConfig:
    """Embedding backend — intentionally swappable (spec §63)."""

    model: str  # placeholder until a local embedding model is chosen
    endpoint: Optional[str]  # null until the serving endpoint exists
    dimensions: Optional[int]  # null until known from the model

    @classmethod
    def from_dict(cls, data: dict) -> "EmbeddingsConfig":
        where = "embeddings"
        return cls(
            model=_get_str(data, "model", where),
            endpoint=_get_optional_str(data, "endpoint", where),
            dimensions=_get_optional_int(data, "dimensions", where),
        )


@dataclass(frozen=True)
class DiversityConfig:
    """Source-diversity caps for context assembly (spec §20)."""

    max_paragraphs_per_source: int  # soft cap on paragraphs from one source document

    @classmethod
    def from_dict(cls, data: dict) -> "DiversityConfig":
        return cls(max_paragraphs_per_source=_get_int(data, "max_paragraphs_per_source", "diversity"))


# ---------------------------------------------------------------------------
# Top-level Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Config:
    """Validated view of the workspace-root config.json."""

    llm: LLMConfig
    experiment: ExperimentConfig
    retrieval: RetrievalConfig
    agentic_retrieval: AgenticRetrievalConfig
    embeddings: EmbeddingsConfig
    diversity: DiversityConfig
    path: Optional[Path] = field(default=None, compare=False)  # file this was loaded from
    raw: dict = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def from_dict(cls, data: Any, path: Optional[Path] = None) -> "Config":
        if not isinstance(data, dict):
            raise ConfigError(
                f"invalid config root in {CONFIG_FILENAME}: expected a JSON object, got {_type_name(data)}"
            )
        cfg = cls(
            llm=LLMConfig.from_dict(_get_dict(data, "llm", "")),
            experiment=ExperimentConfig.from_dict(_get_dict(data, "experiment", "")),
            retrieval=RetrievalConfig.from_dict(_get_dict(data, "retrieval", "")),
            agentic_retrieval=AgenticRetrievalConfig.from_dict(
                _get_dict(data, "agentic_retrieval", "")
            ),
            embeddings=EmbeddingsConfig.from_dict(_get_dict(data, "embeddings", "")),
            diversity=DiversityConfig.from_dict(_get_dict(data, "diversity", "")),
            path=path,
            raw=data,
        )
        _validate_cross_field(cfg)
        return cfg

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Look up an (possibly not-yet-typed) key in the raw config by dotted path.

        Returns ``default`` when any segment is missing — intended for forward-
        compatible access to sections added in later phases (e.g. "evaluation").
        Raises ConfigError if a present segment has the wrong type for traversal.
        """
        node: Any = self.raw
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


def _validate_cross_field(cfg: Config) -> None:
    """Sanity checks that span multiple keys; each failure names the key(s)."""

    def check(condition: bool, message: str) -> None:
        if not condition:
            raise ConfigError(f"invalid config: {message}")

    llm = cfg.llm
    check(bool(llm.base_url.strip()), "llm.base_url must be a non-empty URL")
    check(
        llm.default_model in llm.models or bool(llm.default_model.strip()),
        f"llm.default_model '{llm.default_model}' is neither a known tier "
        f"({sorted(llm.models)}) nor a usable model id",
    )
    check(llm.timeout_seconds > 0, "llm.timeout_seconds must be > 0")
    check(llm.max_retries >= 0, "llm.max_retries must be >= 0")
    check(
        llm.retry_backoff_base_seconds > 0,
        "llm.retry_backoff_base_seconds must be > 0",
    )
    check(
        llm.retry_max_delay_seconds >= llm.retry_backoff_base_seconds,
        "llm.retry_max_delay_seconds must be >= llm.retry_backoff_base_seconds",
    )

    s = llm.sampling
    check(s.temperature >= 0, "llm.sampling.temperature must be >= 0")
    check(0 < s.top_p <= 1, "llm.sampling.top_p must be in (0, 1]")
    check(s.top_k >= 1, "llm.sampling.top_k must be >= 1")
    check(s.max_tokens >= 1, "llm.sampling.max_tokens must be >= 1")

    r = cfg.retrieval
    check(0 <= r.similarity_threshold <= 1, "retrieval.similarity_threshold must be in [0, 1]")
    check(r.neighbor_window >= 0, "retrieval.neighbor_window must be >= 0")
    check(r.max_context_tokens > 0, "retrieval.max_context_tokens must be > 0")
    check(r.paragraph_max_chars > 0, "retrieval.paragraph_max_chars must be > 0")
    check(
        0 <= r.paragraph_overlap_chars < r.paragraph_max_chars,
        "retrieval.paragraph_overlap_chars must be >= 0 and strictly smaller than "
        "retrieval.paragraph_max_chars",
    )
    for name in ("book_knee", "chapter_knee", "topic_knee", "paragraph_knee"):
        knee = getattr(r, name)
        check(
            1 <= knee.minimum <= knee.maximum <= knee.candidate_k,
            f"retrieval.{name} must satisfy 1 <= minimum ({knee.minimum}) <= maximum "
            f"({knee.maximum}) <= candidate_k ({knee.candidate_k})",
        )

    check(
        cfg.agentic_retrieval.max_tool_calls >= 0,
        "agentic_retrieval.max_tool_calls must be >= 0",
    )
    check(
        cfg.diversity.max_paragraphs_per_source >= 1,
        "diversity.max_paragraphs_per_source must be >= 1",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def default_config_path() -> Path:
    """Workspace-root config.json path, resolved relative to this file.

    Anchored at ``<root>/dsa_mentor/config.py`` so it is independent of the
    current working directory.
    """
    return Path(__file__).resolve().parent.parent / CONFIG_FILENAME


def load_config(path: Optional[str | Path] = None) -> Config:
    """Load and validate config.json (from ``path``, or the workspace root).

    Raises:
        ConfigError: file missing, unreadable, invalid JSON, or failed validation.
            Every message names the offending key/file so it is actionable.
    """
    config_path = Path(path).resolve() if path is not None else default_config_path()
    if not config_path.is_file():
        raise ConfigError(
            f"config file not found: {config_path} "
            f"(expected the project's single global config at the workspace root)"
        )
    try:
        text = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"could not read config file {config_path}: {exc}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConfigError(
            f"invalid JSON in {config_path} at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc
    return Config.from_dict(data, path=config_path)
