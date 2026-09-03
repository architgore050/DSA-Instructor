"""System prompt generation for the DSA instructor (spec §26–§27).

Provides ``get_system_prompt()`` which assembles the system message
according to the RAG and tool availability flags.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Instructor behaviour guidelines (spec §27)
# ---------------------------------------------------------------------------

_INSTRUCTOR_GUIDELINES = (
    "You are a DSA instructor. Your goal is to teach, not merely to produce answers.\n"
    "\n"
    "When helping with a problem, follow this structure:\n"
    "1. Identify the structural clue.\n"
    "2. Identify the candidate technique.\n"
    "3. State the invariant or intuition.\n"
    "4. Establish correctness.\n"
    "5. Analyze complexity.\n"
    "6. Consider edge cases.\n"
    "7. Give implementation guidance only when the user asks for it.\n"
    "\n"
    "Do not immediately dump code unless the user explicitly asks for it.\n"
    "Prefer reasoning, intuition, and step-by-step explanation.\n"
)


# ---------------------------------------------------------------------------
# RAG-enabled prompt
# ---------------------------------------------------------------------------

_RAG_PROMPT = (
    "You are a DSA instructor with access to a curated DSA knowledge base. "
    "Use retrieved evidence to ground source-dependent claims. "
    "You may reason using general algorithmic knowledge, but do not pretend that "
    "unsupported claims came from the corpus. Cite retrieved sources when you use them."
)


# ---------------------------------------------------------------------------
# Tool-enabled prompt fragment
# ---------------------------------------------------------------------------

_TOOL_PROMPT = (
    "You have access to a DSA knowledge retrieval tool. "
    "Use it when the supplied evidence is insufficient, when a claim needs "
    "verification, or when the question spans concepts that initial retrieval "
    "does not adequately cover."
)


# ---------------------------------------------------------------------------
# RAG-disabled prompt
# ---------------------------------------------------------------------------

_NO_RAG_PROMPT = (
    "You are a DSA instructor. Answer using your general knowledge of algorithms "
    "and data structures."
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_system_prompt(
    rag_enabled: bool = True,
    tool_enabled: bool = True,
) -> str:
    """Build the system prompt for the DSA instructor.

    Parameters
    ----------
    rag_enabled : bool
        Whether the knowledge-base retrieval context is available.
        When ``False`` the prompt contains no reference to a knowledge base
        or retrieval tool (spec §71–§72).
    tool_enabled : bool
        Whether the model has access to a retrieval tool.
        Ignored when ``rag_enabled=False`` — the tool is unavailable whenever
        RAG is off (spec §72).

    Returns
    -------
    str
        The full system prompt string.
    """
    parts: list[str] = []

    if rag_enabled:
        parts.append(_RAG_PROMPT)
    else:
        parts.append(_NO_RAG_PROMPT)

    parts.append(_INSTRUCTOR_GUIDELINES)

    if rag_enabled and tool_enabled:
        parts.append(_TOOL_PROMPT)

    return "\n\n".join(parts)
