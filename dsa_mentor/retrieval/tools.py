"""Agentic retrieval tool: ``search_knowledge`` (spec §21–§22).

Provides:

- ``search_knowledge_tool(retriever, max_results=8)`` — the OpenAI-compatible
  function-calling schema dict.
- ``create_tool_executor(retriever, max_results_from_config)`` — a
  ``tool_executor(name, args) -> str`` callable wired to the retriever.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

from ..models import Paragraph, RetrievalResult, Topic
from .hierarchy import KneeHierarchicalRetriever

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------


def search_knowledge_tool(
    retriever: KneeHierarchicalRetriever,
    max_results: int = 8,
) -> Dict[str, Any]:
    """Return the OpenAI-compatible tool definition dict for ``search_knowledge``.

    Parameters
    ----------
    retriever : KneeHierarchicalRetriever
        The hierarchical retriever (used to validate the tool is wired to a
        real retriever; the schema itself is stateless).
    max_results : int
        Default maximum number of results when the caller omits the parameter.

    Returns
    -------
    dict
        A tool schema compatible with the OpenAI ``tools`` parameter.
    """
    return {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": (
                "Search the DSA knowledge corpus for relevant evidence. "
                "Returns structured results with source metadata."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query",
                    },
                    "scope": {
                        "type": "string",
                        "enum": ["all", "book", "chapter", "topic"],
                        "description": "Search scope",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results",
                    },
                },
                "required": ["query"],
            },
        },
    }


# ---------------------------------------------------------------------------
# Tool executor factory
# ---------------------------------------------------------------------------


def create_tool_executor(
    retriever: KneeHierarchicalRetriever,
    max_results_from_config: int,
) -> Callable[[str, Dict[str, Any]], str]:
    """Build a ``tool_executor`` for the agentic tool-call loop.

    The returned callable is compatible with :func:`dsa_mentor.llm.run_tool_loop`:

        ``executor(name: str, args: dict) -> str``

    Parameters
    ----------
    retriever : KneeHierarchicalRetriever
        The knee-aware hierarchical retriever providing the search methods.
    max_results_from_config : int
        Default max results from config (typically
        ``config.agentic_retrieval.max_tool_calls`` or a dedicated value).

    Returns
    -------
    callable
        ``(name, args) -> str`` executor.
    """
    manager = retriever.manager

    def executor(name: str, args: Dict[str, Any]) -> str:
        if name != "search_knowledge":
            return f"ERROR: Unknown tool '{name}'"

        # --- argument extraction & validation ---
        if not isinstance(args, dict):
            return "ERROR: Missing or invalid argument: args must be a JSON object"

        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            return "ERROR: Missing or invalid argument: 'query' must be a non-empty string"

        scope = args.get("scope", "all")
        if scope not in ("all", "book", "chapter", "topic"):
            return (
                f"ERROR: Missing or invalid argument: 'scope' must be one of "
                f"{{'all', 'book', 'chapter', 'topic'}} (got {scope!r})"
            )

        max_results = args.get("max_results")
        if max_results is not None:
            if not isinstance(max_results, int) or max_results < 1:
                return (
                    "ERROR: Missing or invalid argument: 'max_results' must be "
                    "a positive integer"
                )
        else:
            max_results = max_results_from_config

        # --- dispatch by scope ---
        try:
            if scope == "all":
                return _format_all_scope(retriever, query, max_results)
            elif scope == "book":
                return _format_book_scope(retriever, query, max_results)
            elif scope == "chapter":
                return _format_chapter_scope(retriever, query, max_results)
            elif scope == "topic":
                return _format_topic_scope(retriever, query, max_results)
        except Exception as exc:
            logger.warning("search_knowledge executor error: %s", exc)
            return f"ERROR: Tool execution failed: {exc}"

    return executor


# ---------------------------------------------------------------------------
# Scope-specific formatting helpers
# ---------------------------------------------------------------------------


def _format_all_scope(
    retriever: KneeHierarchicalRetriever,
    query: str,
    max_results: int,
) -> str:
    """Execute full-pipeline retrieval and format as structured text."""
    result = retriever.retrieve(query)
    lines = [f'Search Results for: "{query}"', ""]

    if not result.paragraphs:
        lines.append("(no results)")
        return "\n".join(lines)

    for i, para in enumerate(result.paragraphs[:max_results], 1):
        book = getattr(para, "book_id", "Unknown") or "Unknown"
        chapter = getattr(para, "chapter_id", "") or ""
        topic = getattr(para, "topic_id", "") or ""
        parts = [p for p in (book, chapter, topic) if p]
        heading = " / ".join(parts) if parts else "Unknown"
        content = para.content or ""
        snippet = content[:200] + ("…" if len(content) > 200 else "")
        sim = getattr(para, "similarity", 0.0)
        lines.append(
            f"[{i}] {heading} \u2014 similarity: {sim:.2f}"
        )
        lines.append(f"    {snippet}")

    return "\n".join(lines)


def _format_book_scope(
    retriever: KneeHierarchicalRetriever,
    query: str,
    max_results: int,
) -> str:
    """Search book index and format results."""
    results = retriever.manager.search_book(query, k=max_results)
    lines = [f'Search Results for: "{query}"', ""]

    if not results:
        lines.append("(no results)")
        return "\n".join(lines)

    for i, (book, score) in enumerate(results[:max_results], 1):
        title = getattr(book, "title", "Unknown") or "Unknown"
        bid = getattr(book, "id", "") or ""
        lines.append(f"[{i}] {bid} \u2014 {title} \u2014 similarity: {score:.2f}")

    return "\n".join(lines)


def _format_chapter_scope(
    retriever: KneeHierarchicalRetriever,
    query: str,
    max_results: int,
) -> str:
    """Search chapter index and format results."""
    results = retriever.manager.search_chapter(query, book_ids=[], k=max_results)
    lines = [f'Search Results for: "{query}"', ""]

    if not results:
        lines.append("(no results)")
        return "\n".join(lines)

    for i, (ch, score) in enumerate(results[:max_results], 1):
        title = getattr(ch, "title", "Unknown") or "Unknown"
        cid = getattr(ch, "id", "") or ""
        lines.append(f"[{i}] {cid} \u2014 {title} \u2014 similarity: {score:.2f}")

    return "\n".join(lines)


def _format_topic_scope(
    retriever: KneeHierarchicalRetriever,
    query: str,
    max_results: int,
) -> str:
    """Search topic index and format results."""
    results = retriever.manager.search_topic(query, chapter_ids=[], k=max_results)
    lines = [f'Search Results for: "{query}"', ""]

    if not results:
        lines.append("(no results)")
        return "\n".join(lines)

    for i, (topic, score) in enumerate(results[:max_results], 1):
        title = getattr(topic, "title", "Unknown") or "Unknown"
        tid = getattr(topic, "id", "") or ""
        lines.append(f"[{i}] {tid} \u2014 {title} \u2014 similarity: {score:.2f}")

    return "\n".join(lines)
