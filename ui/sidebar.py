"""
Sidebar UI components.

Renders the control panel: knowledge source inputs, retrieval
settings, sync button, status display, and action buttons.

Returns a dict of all user selections so app.py can act on them.
"""

import streamlit as st

from config import config


def render_sidebar() -> dict:
    """Render the sidebar and return all user selections.

    Returns:
        A dict with keys:
            uploaded_file, yt_url, search_type, top_k, top_n,
            use_reranker, use_hybrid, process_button,
            clear_kb, clear_chat
    """
    with st.sidebar:
        st.header("🎛️ Control Panel")

        # ── Knowledge Sources ──────────────────────────────────
        st.subheader("📂 Knowledge Sources")
        uploaded_file = st.file_uploader(
            "Upload Research PDF", type="pdf"
        )
        yt_url = st.text_input("YouTube URL")
        st.divider()

        # ── Retrieval Settings ─────────────────────────────────
        st.subheader("⚙️ Retrieval Settings")

        search_type = st.selectbox(
            "Search Strategy",
            ["mmr", "similarity"],
            index=0,
            help=(
                "**MMR** (Maximal Marginal Relevance) reduces redundancy "
                "by diversifying results. **Similarity** returns the "
                "closest matches by cosine distance."
            ),
        )

        top_k = st.slider(
            "Initial candidates (k)",
            min_value=2,
            max_value=15,
            value=config.RETRIEVER_K,
            help="Candidates retrieved before reranking.",
        )

        top_n = st.slider(
            "Final chunks after reranking",
            min_value=1,
            max_value=10,
            value=min(config.RERANKER_TOP_N, top_k),
            help="Chunks kept after cross-encoder re-scoring.",
        )

        use_reranker = st.toggle(
            "⚡ Cross-encoder reranking",
            value=config.RERANKER_ENABLED,
            help=(
                "Re-score candidates with a cross-encoder for higher precision. "
                "⚠️ Adds ~1-2s latency per query on CPU. "
                "Turn ON only when answer quality matters more than speed."
            ),
        )

        use_hybrid = st.toggle(
            "Hybrid search (BM25 + Vector)",
            value=True,
            help=(
                "Combine keyword (BM25) and semantic (vector) search. "
                "Disable for pure semantic search."
            ),
        )

        st.divider()
        process_button = st.button(
            "⚡ Sync Knowledge Base", use_container_width=True
        )

        # ── Status Display ─────────────────────────────────────
        clear_kb = False
        clear_chat = False

        if st.session_state.get("vector_store"):
            st.markdown(
                "<span class='status-badge status-active'>"
                "● Knowledge Base Active</span>",
                unsafe_allow_html=True,
            )

            chunk_count = st.session_state.get("chunk_count", 0)
            st.caption(f"📊 {chunk_count} chunks indexed")

            sources = st.session_state.get("indexed_sources", [])
            if sources:
                st.markdown("**Indexed Sources:**")
                for src in sources:
                    st.markdown(f"- {src}")

            col1, col2 = st.columns(2)
            with col1:
                clear_kb = st.button(
                    "🗑️ Clear KB", use_container_width=True
                )
            with col2:
                clear_chat = st.button(
                    "🧹 Clear Chat", use_container_width=True
                )
        else:
            st.markdown(
                "<span class='status-badge status-inactive'>"
                "○ Knowledge Base Inactive</span>",
                unsafe_allow_html=True,
            )

    return {
        "uploaded_file": uploaded_file,
        "yt_url": yt_url,
        "search_type": search_type,
        "top_k": top_k,
        "top_n": top_n,
        "use_reranker": use_reranker,
        "use_hybrid": use_hybrid,
        "process_button": process_button,
        "clear_kb": clear_kb,
        "clear_chat": clear_chat,
    }
