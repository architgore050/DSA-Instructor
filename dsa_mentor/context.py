"""Context builder — assembles the full LLM message list (spec §59).

Provides ``ContextBuilder.build()`` which takes a
:class:`dsa_mentor.models.RetrievalResult` and produces the OpenAI-style
message list consumed by :class:`dsa_mentor.llm.LLMClient.chat`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from collections import OrderedDict

from .models import Paragraph, RetrievalResult, Topic
from .prompts import get_system_prompt


# ---------------------------------------------------------------------------
# ContextBuilder
# ---------------------------------------------------------------------------


class ContextBuilder:
    """Assemble LLM messages from a retrieval result (spec §59).

    The resulting message list has the structure:

        [system]  → system prompt
        [user]    → QUESTION + CONVERSATION CONTEXT + RETRIEVED KNOWLEDGE + INSTRUCTIONS
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        retrieval_result: RetrievalResult,
        user_query: str,
        conversation_context: str = "",
        rag_enabled: bool = True,
    ) -> list[dict]:
        """Build the full message list for the LLM.

        Parameters
        ----------
        retrieval_result : RetrievalResult
            The output of a retriever (knee-aware or flat).
        user_query : str
            The user's current question.
        conversation_context : str
            Optional summary of prior conversation turns.
        rag_enabled : bool
            When ``False`` only the system prompt and user query are returned
            (no retrieved knowledge section).

        Returns
        -------
        list[dict]
            OpenAI-style message list (``role`` + ``content`` dicts).
        """
        messages: list[dict] = []

        # System message
        messages.append({
            "role": "system",
            "content": get_system_prompt(
                rag_enabled=rag_enabled,
                tool_enabled=rag_enabled,  # tool unavailable when RAG is off
            ),
        })

        if not rag_enabled:
            # RAG OFF: just the question, no retrieved knowledge
            messages.append({
                "role": "user",
                "content": self._build_user_message(user_query, conversation_context, rag_enabled=False),
            })
            return messages

        # RAG ON: question + conversation context + retrieved knowledge + instructions
        messages.append({
            "role": "user",
            "content": self._build_user_message(
                user_query,
                conversation_context,
                rag_enabled=True,
                retrieval_result=retrieval_result,
            ),
        })

        return messages

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_user_message(
        self,
        user_query: str,
        conversation_context: str,
        rag_enabled: bool,
        retrieval_result: Optional[RetrievalResult] = None,
    ) -> str:
        """Build the user message content string."""
        parts: list[str] = []

        # QUESTION
        parts.append(f"QUESTION: {user_query}")

        # CONVERSATION CONTEXT
        if conversation_context:
            parts.append(f"CONVERSATION CONTEXT: {conversation_context}")

        if rag_enabled and retrieval_result is not None:
            parts.append("RETRIEVED KNOWLEDGE:")
            parts.append(self._format_retrieved(retrieval_result))

        # INSTRUCTIONS
        parts.append(
            "INSTRUCTIONS:\n"
            "- Reason rigorously about algorithms and data structures.\n"
            "- Use retrieved evidence to ground source-dependent claims.\n"
            "- You may also use your general knowledge of algorithms.\n"
            "- Do not fabricate or invent sources, page numbers, or citations.\n"
            "- Cite retrieved sources when you use them.\n"
        )

        return "\n\n".join(parts)

    @staticmethod
    def _format_retrieved(result: RetrievalResult) -> str:
        """Format the retrieved knowledge section (topics + subtopic chunks).

        Paragraphs are aggregated by subtopic_id into subtopic-level chunks
        so the LLM receives coherent units rather than fragmented paragraphs.
        """
        sections: list[str] = []

        # Expanded topics (spec §7 — full topic text)
        if result.expanded_topics:
            for topic in result.expanded_topics:
                topic_line = ContextBuilder._topic_heading(topic)
                if isinstance(topic, Topic) and topic.full_text:
                    sections.append(f"[TOPIC] {topic_line}\n{topic.full_text}")
                elif hasattr(topic, "full_text") and getattr(topic, "full_text", None):
                    sections.append(f"[TOPIC] {topic_line}\n{topic.full_text}")
                else:
                    sections.append(f"[TOPIC] {topic_line}\n{getattr(topic, 'content', '') or ''}")

        # Regular topics (without full_text expansion)
        if result.topics:
            for topic in result.topics:
                topic_line = ContextBuilder._topic_heading(topic)
                full_text = None
                if isinstance(topic, Topic):
                    full_text = topic.full_text
                if full_text is None:
                    full_text = getattr(topic, "full_text", None)
                if full_text and full_text.strip():
                    sections.append(f"[TOPIC] {topic_line}\n{full_text}")

        # Paragraphs — aggregate by subtopic into subtopic-level chunks
        if result.paragraphs:
            subtopic_chunks: OrderedDict[str, list] = OrderedDict()
            for para in result.paragraphs:
                st_id = getattr(para, "subtopic_id", None) or getattr(para, "hierarchical_chunk_id", None)
                if st_id:
                    subtopic_chunks.setdefault(st_id, []).append(para)

            for st_id, para_list in subtopic_chunks.items():
                # Derive subtopic title from first paragraph
                first_para = para_list[0]
                st_title = getattr(first_para, "title", "Subtopic")
                st_book = getattr(first_para, "book_id", "")
                st_chapter = getattr(first_para, "chapter_id", "")
                st_topic = getattr(first_para, "topic_id", "")
                st_parts = [p for p in (st_book, st_chapter, st_topic, st_title) if p]
                heading = " / ".join(st_parts)

                # Combine all paragraph contents within this subtopic
                para_contents = []
                for p in para_list:
                    content = p.content if isinstance(p, Paragraph) else getattr(p, "content", "")
                    if content:
                        para_contents.append(content)

                chunk_text = "\n\n".join(para_contents)
                sections.append(f"[SUBTOPIC] {heading}\n{chunk_text}")

        return "\n\n".join(sections)

    @staticmethod
    def _topic_heading(topic: Any) -> str:
        """Format a topic heading: book / chapter / topic."""
        book = getattr(topic, "book_id", None) or getattr(topic, "title", "Unknown")
        chapter = getattr(topic, "chapter_id", None) or ""
        title = getattr(topic, "title", "Unknown")
        parts = [p for p in (book, chapter, title) if p]
        return " / ".join(parts)

    @staticmethod
    def _paragraph_heading(para: Any) -> str:
        """Format a paragraph heading: book / chapter / topic / p.page."""
        book = getattr(para, "book_id", None) or ""
        chapter = getattr(para, "chapter_id", None) or ""
        topic = getattr(para, "topic_id", None) or ""
        page = getattr(para, "page_number", None)
        parts = [p for p in (book, chapter, topic) if p]
        if page is not None:
            parts.append(f"p.{page}")
        return " / ".join(parts) if parts else "Unknown"

    # ------------------------------------------------------------------
    # Token estimation
    # ------------------------------------------------------------------

    @staticmethod
    def estimate_tokens(messages: list[dict]) -> int:
        """Rough token estimator for a message list.

        Uses the heuristic: total character count / 4 ≈ token count for
        English text.

        Parameters
        ----------
        messages : list[dict]
            OpenAI-style message list.

        Returns
        -------
        int
            Estimated token count.
        """
        total_chars = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total_chars += len(content)
        return total_chars // 4


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def build_context(
    retrieval_result: RetrievalResult,
    user_query: str,
    conversation_context: str = "",
    rag_enabled: bool = True,
) -> list[dict]:
    """Build LLM messages from a retrieval result.

    Convenience wrapper around :class:`ContextBuilder.build`.

    Parameters
    ----------
    retrieval_result : RetrievalResult
        Output of a retriever.
    user_query : str
        The user's question.
    conversation_context : str
        Optional prior conversation summary.
    rag_enabled : bool
        Whether RAG context is available.

    Returns
    -------
    list[dict]
        OpenAI-style message list.
    """
    return ContextBuilder().build(
        retrieval_result=retrieval_result,
        user_query=user_query,
        conversation_context=conversation_context,
        rag_enabled=rag_enabled,
    )
