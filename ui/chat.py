"""
Chat interface components.

Handles rendering of chat history, source display, and metadata
extraction for history storage.
"""

import streamlit as st
from langchain_core.documents import Document


def render_chat_history() -> None:
    """Render all past chat turns using st.chat_message.

    Reads from st.session_state["chat_history"] — a list of dicts
    with keys: question, answer, sources.
    """
    for turn in st.session_state.get("chat_history", []):
        with st.chat_message("user"):
            st.write(turn["question"])

        with st.chat_message("assistant"):
            st.write(turn["answer"])

            sources = turn.get("sources", [])
            if sources:
                with st.expander(
                    f"📚 Sources ({len(sources)} chunks)"
                ):
                    _render_source_list(sources)


def display_sources(docs: list[Document]) -> None:
    """Display retrieved documents in an expander below the answer.

    Args:
        docs: The actual Document objects returned by the retriever.
    """
    with st.expander(f"📚 Sources ({len(docs)} chunks)"):
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "Unknown")
            doc_type = doc.metadata.get("type", "unknown")
            page = doc.metadata.get("page", "")
            page_str = f", p.{page}" if page else ""

            preview = (
                doc.page_content[:250] + "…"
                if len(doc.page_content) > 250
                else doc.page_content
            )

            st.markdown(
                f"<div class='source-box'>"
                f"<b>Source {i + 1}:</b> {source}{page_str} "
                f"[{doc_type}]<br>"
                f"<small>{preview}</small></div>",
                unsafe_allow_html=True,
            )


def build_source_metadata(docs: list[Document]) -> list[dict]:
    """Extract display-friendly metadata from retrieved documents.

    This is stored in chat history so sources can be re-rendered
    without holding references to the original Document objects.

    Args:
        docs: Retrieved Document objects.

    Returns:
        A list of dicts with keys: name, preview.
    """
    sources = []
    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        doc_type = doc.metadata.get("type", "unknown")
        page = doc.metadata.get("page", "")
        page_str = f", p.{page}" if page else ""

        preview = (
            doc.page_content[:250] + "…"
            if len(doc.page_content) > 250
            else doc.page_content
        )
        sources.append(
            {
                "name": f"{source}{page_str} [{doc_type}]",
                "preview": preview,
            }
        )
    return sources


# ── Internal helpers ───────────────────────────────────────────

def _render_source_list(sources: list[dict]) -> None:
    """Render a list of source metadata dicts as styled HTML boxes."""
    for i, src in enumerate(sources):
        st.markdown(
            f"<div class='source-box'>"
            f"<b>Source {i + 1}:</b> {src['name']}<br>"
            f"<small>{src['preview']}</small></div>",
            unsafe_allow_html=True,
        )
