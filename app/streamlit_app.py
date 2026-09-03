"""DSA Mentor — Streamlit chat UI with retrieval inspector.

Implements spec §28 (interaction modes), §29 (chat UI layout), §30-§31
(retrieval inspector with expandable tree showing books/chapters/topics/
paragraphs with knee info).

Run with: ``streamlit run app/streamlit_app.py``
"""

from __future__ import annotations

import json
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
    for sub in ("books", "chapters", "topics", "paragraphs"):
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

def _render_retrieval_inspector(result: RetrievalResult) -> None:
    """Render the retrieval inspector as a collapsible section."""
    with st.expander("🔍 Retrieval Inspector", expanded=False):
        st.caption(f"**Query:** {result.query}")
        st.divider()

        # --- Books ---
        if result.books:
            st.subheader("Books", divider=True)
            for i, book in enumerate(result.books, 1):
                title = getattr(book, "title", "Unknown")
                bid = getattr(book, "id", "")
                score = getattr(book, "similarity", 0.0)
                st.markdown(
                    f"{i}. **{title}** &nbsp; {score:.2f} "
                    f"{'✓ selected' if i else '✗ excluded'}"
                )
            knee = result.knees.get("book")
            if knee:
                st.caption(
                    f"Knee: selected through rank {knee.knee_index} "
                    f"of {knee.candidate_k} candidates"
                )
        else:
            st.subheader("Books", divider=True)
            st.caption("No books retrieved.")

        st.divider()

        # --- Chapters ---
        if result.chapters:
            st.subheader("Chapters", divider=True)
            for i, ch in enumerate(result.chapters, 1):
                title = getattr(ch, "title", "Unknown")
                cid = getattr(ch, "id", "")
                score = getattr(ch, "similarity", 0.0)
                st.markdown(f"{i}. **{title}** &nbsp; {score:.2f} ✓ selected")
            knee = result.knees.get("chapter")
            if knee:
                st.caption(
                    f"Knee: selected through rank {knee.knee_index} "
                    f"of {knee.candidate_k} candidates"
                )
        else:
            st.subheader("Chapters", divider=True)
            st.caption("No chapters retrieved.")

        st.divider()

        # --- Topics ---
        if result.topics:
            st.subheader("Topics", divider=True)
            for i, topic in enumerate(result.topics, 1):
                title = getattr(topic, "title", "Unknown")
                tid = getattr(topic, "id", "")
                score = getattr(topic, "similarity", 0.0)
                st.markdown(f"{i}. **{title}** &nbsp; {score:.2f} ✓ selected")
            knee = result.knees.get("topic")
            if knee:
                st.caption(
                    f"Knee: selected through rank {knee.knee_index} "
                    f"of {knee.candidate_k} candidates"
                )
        else:
            st.subheader("Topics", divider=True)
            st.caption("No topics retrieved.")

        st.divider()

        # --- Paragraphs ---
        if result.paragraphs:
            st.subheader("Paragraphs", divider=True)
            for i, para in enumerate(result.paragraphs, 1):
                title = getattr(para, "title", f"Paragraph {i}")
                score = getattr(para, "similarity", 0.0)
                book_id = getattr(para, "book_id", "")
                chapter_id = getattr(para, "chapter_id", "")
                topic_id = getattr(para, "topic_id", "")
                citation_parts = [p for p in (book_id, chapter_id, topic_id) if p]
                citation = " / ".join(citation_parts) if citation_parts else "Unknown"
                st.markdown(
                    f"{i}. **{title}** &nbsp; {score:.2f} ✓ selected "
                    f"[_{citation}_]"
                )
            knee = result.knees.get("paragraph")
            if knee:
                st.caption(
                    f"Knee: selected through rank {knee.knee_index} "
                    f"of {knee.candidate_k} candidates"
                )
        else:
            st.subheader("Paragraphs", divider=True)
            st.caption("No paragraphs retrieved.")

        st.divider()

        # --- Tool calls ---
        if result.tool_calls:
            st.subheader("Tool Calls", divider=True)
            for tc in result.tool_calls:
                tc_dict = tc.to_dict() if hasattr(tc, "to_dict") else asdict(tc)
                st.markdown(f"**Tool:** `search_knowledge`")
                st.markdown(f"**Query:** {tc_dict.get('query', '')}")
                results = tc_dict.get("results", [])
                if isinstance(results, list):
                    for r in results[:5]:
                        if isinstance(r, dict):
                            st.caption(
                                f"- {r.get('title', r.get('heading', 'Result'))}"
                            )
                st.divider()

        # --- Knee summary ---
        if result.knees:
            st.subheader("Knee Summary", divider=True)
            for level, kd in result.knees.items():
                st.caption(
                    f"{level}: selected through rank {kd.knee_index} "
                    f"of {kd.candidate_k} candidates "
                    f"(threshold={kd.threshold:.3f})"
                )

        # --- Context budget ---
        st.caption(f"Context tokens: {result.context_tokens}")


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
# Message formatting
# ===================================================================

def _format_message(content: str) -> str:
    """Format message content for proper newline rendering in Streamlit.

    Wraps content in a preformatted div that preserves whitespace and newlines,
    while still allowing Markdown rendering for other syntax.
    """
    if not content:
        return ""
    # Use HTML pre tag to preserve whitespace/newlines, then render as HTML
    return f"<pre style='margin:0;padding:0;font-family:inherit;white-space:pre-wrap;word-wrap:break-word;'>{content}</pre>"


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
                assistant_placeholder.markdown(_format_message(full_content))
        except Exception as exc:
            error_msg = f"Error: {exc}"
            assistant_placeholder.markdown(_format_message(error_msg))
            streaming_content = error_msg

        st.session_state.messages.append({"role": "assistant", "content": streaming_content})
        st.session_state.conversation_history.append((user_query, streaming_content))
        st.session_state.retrieval_result = None
        return

    # RAG ON
    if not retriever:
        st.error(
            "Retriever not available. Please build the index first by running:\n"
            "`python -m dsa_mentor.ingestion.build --index`"
        )
        st.session_state.messages.pop()
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
        except Exception as exc:
            st.error(f"LLM call failed: {exc}")
            final_content = f"Error: {exc}"

    # Step 4: Stream the final answer
    assistant_placeholder = st.empty()
    streaming_content = ""

    # Stream the final content character by character for UX
    for i in range(0, len(final_content), 1):
        chunk = final_content[:i+1]
        streaming_content = chunk
        assistant_placeholder.markdown(_format_message(chunk))
        time.sleep(0.002)  # small delay for smooth streaming effect

    # Store final results
    st.session_state.messages.append({"role": "assistant", "content": streaming_content})
    st.session_state.conversation_history.append((user_query, streaming_content))
    st.session_state.retrieval_result = retrieval_result


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
                st.markdown(_format_message(content))
        elif role == "assistant":
            with st.chat_message("assistant"):
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
        st.rerun()


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
            "Run `python -m dsa_mentor.ingestion.build --index` to build the knowledge base and FAISS indices.\n\n"
            "Until then, RAG retrieval is disabled."
        )

    # Render sidebar and capture values
    mode, rag_enabled, model_tier = _render_sidebar(config)
    st.session_state._sidebar_values = (mode, rag_enabled, model_tier)

    # Render chat
    _render_chat_area()


if __name__ == "__main__":
    main()
