"""DSA Mentor — Streamlit chat UI with retrieval inspector.

Implements spec §28 (interaction modes), §29 (chat UI layout), §30-§31
(retrieval inspector with expandable tree showing books/chapters/topics/
paragraphs with knee info).

Run with: ``streamlit run app/streamlit_app.py``
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so ``dsa_mentor`` imports work
# regardless of CWD.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dsa_mentor.config import Config, load_config
from dsa_mentor.context import ContextBuilder
from dsa_mentor.llm import LLMClient, ToolLoopResult
from dsa_mentor.models import (
    Book,
    Chapter,
    KneeData,
    Paragraph,
    RetrievalResult,
    Topic,
    ToolCall,
)
from dsa_mentor.retrieval.hierarchy import KneeHierarchicalRetriever
from dsa_mentor.retrieval.tools import create_tool_executor, search_knowledge_tool

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_KB_FILE = _PROJECT_ROOT / "knowledge_base.json"
_INDEX_DIR = _PROJECT_ROOT / "index"


# ===================================================================
# Helpers — config, index building, retriever wiring
# ===================================================================

def _load_config() -> Config:
    try:
        return load_config()
    except Exception as exc:
        st.error(f"Failed to load config.json: {exc}")
        st.stop()


def _index_exists() -> bool:
    """Check whether the multi-index directory has been built (FAISS files present)."""
    if not _INDEX_DIR.is_dir():
        return False
    # Check for actual FAISS index files (not just empty JSON shells)
    for sub in ("books", "chapters", "topics", "subtopics", "paragraphs"):
        faiss_path = _INDEX_DIR / sub / "index.faiss"
        if not faiss_path.exists():
            return False
    return True


def _load_retriever(cfg: Config) -> Optional[KneeHierarchicalRetriever]:
    """Load the multi-index and return a KneeHierarchicalRetriever.
    
    Returns None if index doesn't exist.
    """
    if not _index_exists():
        return None
    
    import importlib
    multi_mod = importlib.import_module("dsa_mentor.index.multi")
    embed_mod = importlib.import_module("dsa_mentor.embeddings")
    
    index_path = str(_INDEX_DIR)
    emb_client = embed_mod.EmbeddingClient(cfg)
    mgr = multi_mod.MultiIndexManager.load(index_path, embedding_client=emb_client)
    retriever = KneeHierarchicalRetriever(
        multi_index_manager=mgr,
        embedding_client=emb_client,
        config=cfg,
    )
    return retriever


# ===================================================================
# Mode-specific system prompt fragments
# ===================================================================

_HINT_SYSTEM_EXTRA = (
    "\n\n**Hint mode:** The student is working on a problem. "
    "Reveal progressively more detail. Start with the most general hint. "
    "On follow-up messages, give increasingly specific hints. "
    "Never give the full solution on the first hint. "
    "Use at most 4 hint levels. End each hint with a question that "
    "encourages the student to think."
)

_EXPLAIN_SYSTEM_EXTRA = (
    "\n\n**Explain mode:** The student has submitted code. Analyze it for: "
    "1) Correctness — does it solve the stated problem? "
    "2) Bugs — identify any logical or syntactic errors. "
    "3) Invariants — state the key invariants the code relies on. "
    "4) Complexity — give time and space complexity with reasoning. "
    "5) Alternatives — suggest better approaches if any. "
    "Quote relevant code lines when critiquing."
)


def _get_mode_system_prompt(cfg: Config, mode: str, rag_enabled: bool) -> str:
    """Return the full system prompt for the current mode."""
    from dsa_mentor.prompts import get_system_prompt

    base = get_system_prompt(rag_enabled=rag_enabled, tool_enabled=rag_enabled)
    if mode == "hint":
        return base + _HINT_SYSTEM_EXTRA
    elif mode == "explain":
        return base + _EXPLAIN_SYSTEM_EXTRA
    return base


# ===================================================================
# Retrieval inspector rendering (spec §30-§31)
# ===================================================================

# Cache for loading knowledge base hierarchy
_retrieval_cache: Dict[str, Any] = {}


def _load_kb_for_inspector() -> Dict[str, Any]:
    """Load knowledge base hierarchy for readable name lookups."""
    cache_key = str(_KB_FILE)
    if cache_key in _retrieval_cache:
        return _retrieval_cache[cache_key]

    if not _KB_FILE.exists():
        _retrieval_cache[cache_key] = {"books": {}, "chapters": {}, "topics": {}, "paragraphs": {}}
        return _retrieval_cache[cache_key]

    try:
        with open(_KB_FILE, "r", encoding="utf-8") as f:
            kb_data = json.load(f)

        books = {b["id"]: b for b in kb_data.get("books", [])}
        chapters = {c["id"]: c for c in kb_data.get("chapters", [])}
        topics = {t["id"]: t for t in kb_data.get("topics", [])}
        paragraphs = {p["id"]: p for p in kb_data.get("paragraphs", [])}

        _retrieval_cache[cache_key] = {
            "books": books,
            "chapters": chapters,
            "topics": topics,
            "paragraphs": paragraphs,
        }
        return _retrieval_cache[cache_key]
    except Exception:
        _retrieval_cache[cache_key] = {"books": {}, "chapters": {}, "topics": {}, "paragraphs": {}}
        return _retrieval_cache[cache_key]


def _get_readable_book_name(book_id: str) -> str:
    """Get a readable book name from ID."""
    if not book_id:
        return "Unknown"
    kb = _load_kb_for_inspector()
    if book_id in kb["books"]:
        return kb["books"][book_id].get("title", book_id)
    # Fallback: try to derive from corpus_id
    corpus_map = {
        "book-001": "Textbooks",
        "book-002": "CP-Algorithms",
        "book-003": "GeeksforGeeks DSA Tutorial",
        "book-004": "JavaScript Algorithms",
        "book-005": "The Algorithms",
        "textbooks": "Textbooks",
        "cp-algorithms": "CP-Algorithms",
        "geeksforgeeks": "GeeksforGeeks DSA Tutorial",
        "javascript-algorithms": "JavaScript Algorithms",
        "thealgorithms": "The Algorithms",
    }
    return corpus_map.get(book_id, book_id.replace("-", " ").title())


def _get_readable_chapter_name(chapter_id: str, book_id: str = "") -> str:
    """Get a readable chapter name from ID."""
    if not chapter_id:
        return "Unknown"
    kb = _load_kb_for_inspector()
    if chapter_id in kb["chapters"]:
        return kb["chapters"][chapter_id].get("title", chapter_id)
    # Try to derive from book + chapter_id
    if book_id:
        book_title = _get_readable_book_name(book_id)
        # Extract directory-like name from chapter_id
        parts = chapter_id.replace("ch-", "").split("-")
        if len(parts) >= 2:
            return f"{book_title} - Chapter {parts[-1]}"
    return chapter_id.replace("ch-", "Chapter ").replace("-", " ")


def _get_readable_topic_name(topic_id: str) -> str:
    """Get a readable topic name from ID."""
    if not topic_id:
        return "Unknown"
    kb = _load_kb_for_inspector()
    if topic_id in kb["topics"]:
        return kb["topics"][topic_id].get("title", topic_id)
    return topic_id.replace("topic-", "Topic ").replace("-", " ")


def _render_retrieval_inspector(result: RetrievalResult) -> None:
    """Render the retrieval inspector as a simple list of retrieved chunks."""
    with st.expander("🔍 Retrieval Results", expanded=False):
        st.caption(f"**Query:** {result.query}")
        st.caption(f"**Context tokens:** {result.context_tokens}")

        if result.knees:
            knee_parts = []
            for level, kd in result.knees.items():
                knee_parts.append(
                    f"{level}: {kd.selected_k}/{kd.candidate_k} "
                    f"(threshold={kd.threshold:.3f})"
                )
            st.caption(f"Knee selection: {' | '.join(knee_parts)}")

        st.divider()

        # --- All chunks as a simple list ---
        has_any = result.books or result.chapters or result.topics or result.paragraphs

        if not has_any:
            st.warning("No chunks retrieved.")
            return

        for i, chunk in enumerate(result.books + result.chapters + result.topics + result.paragraphs, 1):
            if hasattr(chunk, "to_dict"):
                meta = chunk.to_dict()
            elif isinstance(chunk, dict):
                meta = chunk
            else:
                meta = {"id": str(chunk)}

            chunk_type = meta.get("level", "unknown")
            title = meta.get("title", meta.get("id", "Untitled"))
            similarity = meta.get("similarity", None)

            # Build a collapsible block for each chunk's full metadata
            with st.expander(f"#{i} [{chunk_type}] {title}", expanded=False):
                if similarity is not None:
                    st.caption(f"Similarity: {similarity:.4f}")
                for k, v in meta.items():
                    if k == "similarity":
                        continue
                    if k in ("children",) and isinstance(v, list):
                        st.caption(f"{k}: {len(v)} items")
                    elif v is not None:
                        val = str(v)
                        if len(val) > 300:
                            val = val[:300] + "…"
                        st.text(f"{k}: {val}")


# ===================================================================
# Source citation rendering inside assistant messages
# ===================================================================

def _render_source_citations(result: RetrievalResult) -> str:
    """Append source citations to the assistant response text."""
    citations: List[str] = []
    for book in result.books:
        title = getattr(book, "title", "Unknown")
        citations.append(f"[{title}]")
    for ch in result.chapters:
        title = getattr(ch, "title", "Unknown")
        citations.append(f"[{title}]")
    for topic in result.topics:
        title = getattr(topic, "title", "Unknown")
        citations.append(f"[{title}]")
    if citations:
        return "\n\n---\n**Sources:** " + " | ".join(citations[:10])
    return ""


# ===================================================================
# Thinking/reasoning extraction
# ===================================================================

_THINKING_PATTERN = re.compile(
    r'<thinking[^>]*>(.*?)</thinking>', re.DOTALL | re.IGNORECASE
)


def _extract_thinking(content: str) -> Tuple[str, str]:
    """Extract <thinking>...</thinking> block from content.

    Returns (thinking_text, main_content) where thinking_text is empty
    if no thinking block was found.
    """
    if not content:
        return "", content
    match = _THINKING_PATTERN.search(content)
    if match:
        thinking = match.group(1).strip()
        main = content[:match.start()].strip() + content[match.end():].strip()
        return thinking, main
    return "", content


def _format_thinking_html(thinking: str) -> str:
    """Format thinking text as grey, collapsible HTML."""
    if not thinking:
        return ""
    cleaned = thinking
    cleaned = re.sub(r'<br\s*/?>', '\n', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'<[^>]+>', '', cleaned)
    cleaned = cleaned.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    cleaned = cleaned.replace('&quot;', '"').replace('&#39;', "'").replace('&nbsp;', ' ')
    escaped = thinking.replace('"', '&quot;')
    return (
        f'<details open style="margin-bottom:12px">'
        f'<summary style="cursor:pointer;color:#888;font-size:0.85em;font-style:italic">💭 Reasoning</summary>'
        f'<div style="color:#777;font-size:0.9em;line-height:1.5;white-space:pre-wrap">{escaped}</div>'
        f'</details>'
    )


# ===================================================================
# Message formatting
# ===================================================================

def _clean_message(content: str) -> str:
    """Strip HTML tags and convert HTML entities to preserve newlines for markdown rendering.

    The LLM may return HTML-formatted text (e.g. <p>, <br>, <code>, <pre>, <ul>, <li>,
    <b>, <i>, <strong>, <em>, <h1>-<h6>, <div>, <span>). This function:
      1. Strips all HTML tags
      2. Converts <br>/<br/>/<br /> to newlines
      3. Converts &amp; &lt; &gt; &quot; entities back to characters
    """
    if not content:
        return ""
    # Convert <br> variants to newlines BEFORE stripping tags
    content = re.sub(r'<br\s*/?>', '\n', content, flags=re.IGNORECASE)
    # Strip all remaining HTML tags
    content = re.sub(r'<[^>]+>', '', content)
    # Decode common HTML entities
    content = content.replace('&amp;', '&')
    content = content.replace('&lt;', '<')
    content = content.replace('&gt;', '>')
    content = content.replace('&quot;', '"')
    content = content.replace('&#39;', "'")
    content = content.replace('&nbsp;', ' ')
    return content


def _format_message(content: str) -> str:
    """Clean message content for proper rendering via st.markdown().

    Strips HTML tags and normalizes whitespace so st.markdown() renders
    newlines and formatting correctly.
    """
    if not content:
        return ""
    return _clean_message(content)


def _format_user_message(content: str) -> str:
    """Format user message preserving newlines with HTML <br> tags.

    Streamlit's markdown collapses single newlines. We convert them to
    <br> tags to preserve the user's line breaks.
    """
    if not content:
        return ""
    cleaned = _clean_message(content)
    # Convert newlines to <br> to preserve line breaks in markdown
    cleaned = cleaned.replace('\n', '<br>')
    return cleaned


# ===================================================================
# Session state helpers
# ===================================================================

def _init_session_state() -> None:
    """Ensure all required session state keys exist."""
    if "messages" not in st.session_state:
        st.session_state.messages: List[Dict[str, str]] = []
    if "retrieval_result" not in st.session_state:
        st.session_state.retrieval_result: Optional[RetrievalResult] = None
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history: List[Tuple[str, str]] = []
    if "llm_client" not in st.session_state:
        cfg = _load_config()
        st.session_state.llm_client = LLMClient(cfg)
    if "config" not in st.session_state:
        st.session_state.config = _load_config()
    if "retriever" not in st.session_state:
        try:
            st.session_state.retriever = _load_retriever(
                st.session_state.config
            )
        except Exception:
            st.session_state.retriever = None
            logging.getLogger(__name__).exception("Failed to load retriever")


# ===================================================================
# Conversation processing
# ===================================================================

def _build_conversation_context() -> str:
    """Build a compact conversation context string from history."""
    if not st.session_state.conversation_history:
        return ""
    parts = []
    for user_q, assistant_a in st.session_state.conversation_history[-5:]:
        parts.append(f"Q: {user_q}\nA: {assistant_a}")
    return "\n\n".join(parts)


def _process_query(user_query: str, mode: str, model_tier: str, rag_enabled: bool) -> None:
    """Run the full query pipeline and display the result with streaming."""
    client: LLMClient = st.session_state.llm_client
    config: Config = st.session_state.config
    retriever: Optional[KneeHierarchicalRetriever] = st.session_state.retriever

    # Add user message immediately
    st.session_state.messages.append({"role": "user", "content": user_query})

    if not rag_enabled:
        # RAG OFF: direct LLM call with streaming
        system_prompt = _get_mode_system_prompt(config, mode, rag_enabled=False)
        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history
        if st.session_state.conversation_history:
            for uq, ar in st.session_state.conversation_history[-5:]:
                messages.append({"role": "user", "content": uq})
                messages.append({"role": "assistant", "content": ar})

        messages.append({"role": "user", "content": user_query})

        # Create placeholder for streaming assistant response
        assistant_placeholder = st.empty()
        streaming_content = ""

        try:
            for token_delta, full_content in client.chat_stream(messages, model=model_tier):
                streaming_content = full_content
                thinking_text, main_content = _extract_thinking(full_content)
                if thinking_text:
                    display_html = _format_thinking_html(thinking_text) + _format_message(main_content)
                else:
                    display_html = _format_message(main_content)
                assistant_placeholder.markdown(display_html, unsafe_allow_html=True)
        except Exception as exc:
            error_msg = f"Error: {exc}"
            assistant_placeholder.markdown(_format_message(error_msg))
            streaming_content = error_msg

        # Store with thinking tags if present
        full_display = streaming_content
        thinking_text, main_content = _extract_thinking(streaming_content)
        if thinking_text:
            full_display = f"<thinking>{thinking_text}</thinking>{main_content}"
        st.session_state.messages.append({"role": "assistant", "content": full_display})
        st.session_state.conversation_history.append((user_query, main_content if thinking_text else streaming_content))
        st.session_state.retrieval_result = None
        return

    # RAG ON
    if not retriever:
        st.error(
            "Retriever not available. The vector index could not be loaded. "
            "Try rebuilding it:"
        )
        st.code("python -m dsa_mentor.ingestion.build --index", language="bash")
        st.session_state.messages.pop()
        st.session_state._retriever_error = True
        return

    # Step 1: Retrieve
    with st.spinner("Retrieving relevant knowledge …"):
        try:
            retrieval_result = retriever.retrieve(user_query)
        except Exception as exc:
            st.warning(
                f"Retrieval failed ({exc}). Falling back to RAG OFF mode."
            )
            st.session_state.messages.pop()
            _process_query(user_query, mode, model_tier, rag_enabled=False)
            return

    # Step 2: Build conversation context
    conv_context = _build_conversation_context()

    # Step 3: Run agentic tool loop (non-streaming, needs structured responses)
    with st.spinner("Generating response …"):
        try:
            tool_result: ToolLoopResult = client.chat_with_tools(
                retriever=retriever,
                user_query=user_query,
                conversation_context=conv_context,
                rag_enabled=True,
                max_tool_calls=config.agentic_retrieval.max_tool_calls,
                model=model_tier,
            )
            final_content = tool_result.content
            # Fallback: if content is empty but transcript has assistant messages,
            # extract the last non-empty assistant content from the transcript
            if not final_content.strip() and tool_result.transcript:
                for msg in reversed(tool_result.transcript):
                    if msg.get("role") == "assistant" and msg.get("content"):
                        final_content = msg["content"]
                        if final_content.strip():
                            break
        except Exception as exc:
            st.error(f"LLM call failed: {exc}")
            final_content = f"Error: {exc}"

    # Step 4: Stream the final answer
    assistant_placeholder = st.empty()
    streaming_content = ""
    thinking_text, main_content = _extract_thinking(final_content)

    # Stream the final content character by character for UX
    for i in range(0, len(main_content), 1):
        chunk = main_content[:i+1]
        streaming_content = chunk
        # Build display with thinking section if present
        if thinking_text:
            display_html = _format_thinking_html(thinking_text) + _format_message(chunk)
        else:
            display_html = _format_message(chunk)
        assistant_placeholder.markdown(display_html, unsafe_allow_html=True)
        time.sleep(0.002)  # small delay for smooth streaming effect

    # Store final results (keep thinking+main together)
    full_display = streaming_content
    if thinking_text:
        full_display = f"<thinking>{thinking_text}</thinking>{streaming_content}"
    st.session_state.messages.append({"role": "assistant", "content": full_display})
    st.session_state.conversation_history.append((user_query, streaming_content))
    st.session_state.retrieval_result = retrieval_result


# ===================================================================
# Index building from UI
# ===================================================================
# UI Layout
# ===================================================================

def _render_sidebar(cfg: Config) -> Tuple[str, bool, str]:
    """Render the sidebar. Returns (mode, rag_enabled, model_tier)."""
    with st.sidebar:
        st.header("⚙️ Settings", divider=True)

        # Mode selector
        mode = st.selectbox(
            "Mode",
            options=["Learn", "Hint", "Explain"],
            index=0,
            help=(
                "Learn: standard Q&A with retrieval. "
                "Hint: progressive hints for problem solving. "
                "Explain: analyze submitted code."
            ),
        )

        # RAG toggle
        rag_enabled = st.toggle(
            "RAG",
            value=cfg.experiment.rag_enabled,
            help="Toggle retrieval-augmented generation on/off.",
        )

        # Model selector
        model_options = [
            ("Large (qwen3.6-35b)", "large"),
            ("Medium (phi-4-reasoning-plus)", "medium"),
            ("Small (gemma-4-e4b-it)", "small"),
        ]
        model_labels = [o[0] for o in model_options]
        model_tier_map = {label: tier for label, tier in model_options}
        model_tier = st.selectbox(
            "Model",
            options=model_labels,
            index=0,
            help="Select the LLM to use for responses.",
        )
        model_tier = model_tier_map[model_tier]

        # Conversation controls
        st.divider()
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.conversation_history = []
            st.session_state.retrieval_result = None
            st.rerun()

    return mode, rag_enabled, model_tier


def _render_chat_area() -> None:
    """Render the main chat messages and input area."""
    # Display existing messages
    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            with st.chat_message("user"):
                st.markdown(_format_user_message(content), unsafe_allow_html=True)
        elif role == "assistant":
            with st.chat_message("assistant"):
                thinking_text, main_content = _extract_thinking(content)
                if thinking_text:
                    display_html = _format_thinking_html(thinking_text) + _format_message(main_content)
                    st.markdown(display_html, unsafe_allow_html=True)
                else:
                    st.markdown(_format_message(content))
                # Show source citations if retrieval was used
                rr = st.session_state.retrieval_result
                if rr and rr.books:
                    citations = _render_source_citations(rr)
                    if citations:
                        st.caption(citations)

    # Show retrieval inspector below the last assistant message
    rr = st.session_state.retrieval_result
    if rr and (rr.books or rr.chapters or rr.topics or rr.paragraphs or rr.tool_calls):
        _render_retrieval_inspector(rr)

    # Input area
    user_query = st.chat_input("Ask a DSA question…", key="chat_input")
    if user_query and user_query.strip():
        mode, rag_enabled, model_tier = st.session_state._sidebar_values
        _process_query(user_query.strip(), mode, model_tier, rag_enabled)
        if not st.session_state.get("_retriever_error", False):
            st.rerun()
        else:
            st.session_state._retriever_error = False


# ===================================================================
# Main entry point
# ===================================================================

def main() -> None:
    """Run the DSA Mentor Streamlit application."""
    # Page config
    st.set_page_config(
        page_title="DSA Mentor",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Header
    config = _load_config()
    rag_indicator = "🟢 RAG ENABLED" if config.experiment.rag_enabled else "🔴 RAG OFF"
    st.markdown(
        f"<h1>DSA Mentor &nbsp;&nbsp; <span style='font-size:0.7em;color:#888'>{rag_indicator}</span></h1>",
        unsafe_allow_html=True,
    )

    # Init session state (must happen after st.markdown for sidebar)
    _init_session_state()

    # Warn if no index exists
    if not _index_exists():
        st.warning(
            "⚠️ **Vector index not found.** "
            "Build it from the command line before launching the UI:\n\n"
            "```bash\n"
            "python -m dsa_mentor.ingestion.build --index\n"
            "```\n\n"
            "Until then, RAG retrieval is disabled."
        )

    # Render sidebar and capture values
    mode, rag_enabled, model_tier = _render_sidebar(config)
    st.session_state._sidebar_values = (mode, rag_enabled, model_tier)

    # Render chat
    _render_chat_area()


if __name__ == "__main__":
    main()
