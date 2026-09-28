"""
Nexus AI Research Hub — Application Entry Point
=================================================

A modular RAG research assistant with:
  • Hybrid retrieval (BM25 + Vector search)
  • Cross-encoder reranking
  • Streaming LLM responses
  • Conversation memory

This file is intentionally thin — it wires the UI layer to the
Core layer and handles session state.  All business logic lives
in the core/ package.
"""

import streamlit as st

from config import config
from core.document_processor import DocumentProcessor
from core.vector_store import VectorStoreManager
from core.retriever import HybridRetriever
from core.rag_chain import stream_answer
from ui.styles import MAIN_CSS
from ui.sidebar import render_sidebar
from ui.chat import (
    render_chat_history,
    display_sources,
    build_source_metadata,
)
from utils.logger import setup_logger

logger = setup_logger("nexus.app")


# ──────────────────────────────────────────────────────────────
#  PAGE CONFIG  (must be the first Streamlit command)
# ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Nexus AI | Open-Source Research Agent",
    page_icon="🔬",
    layout="wide",
)
st.markdown(MAIN_CSS, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
#  VALIDATE CONFIGURATION
# ──────────────────────────────────────────────────────────────

errors = config.validate()
if errors:
    for err in errors:
        st.error(f"⚙️ Configuration error: {err}")
    st.info(
        "Create a `.env` file from `.env.example` and set the "
        "required values.  See the README for details."
    )
    st.stop()



# ──────────────────────────────────────────────────────────────
#  SESSION STATE INITIALISATION
# ──────────────────────────────────────────────────────────────

_defaults = {
    "vector_store": None,
    "all_chunks": [],
    "chat_history": [],
    "indexed_sources": [],
    "chunk_count": 0,
}
for key, default in _defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ──────────────────────────────────────────────────────────────
#  HEADER
# ──────────────────────────────────────────────────────────────

st.title("🔬 Nexus AI Research Hub")
st.caption(
    "Hybrid RAG with cross-encoder reranking · "
    "Powered by Llama 3.3 & BGE Embeddings"
)


# ──────────────────────────────────────────────────────────────
#  SIDEBAR
# ──────────────────────────────────────────────────────────────

sidebar = render_sidebar()


# ──────────────────────────────────────────────────────────────
#  ACTION HANDLERS
# ──────────────────────────────────────────────────────────────

# -- Clear Knowledge Base --
if sidebar["clear_kb"]:
    VectorStoreManager().clear()
    for key, default in _defaults.items():
        st.session_state[key] = default
    logger.info("Knowledge base and chat history cleared")
    st.rerun()

# -- Clear Chat --
if sidebar["clear_chat"]:
    st.session_state.chat_history = []
    logger.info("Chat history cleared")
    st.rerun()


# ──────────────────────────────────────────────────────────────
#  DOCUMENT PROCESSING  ("Sync Knowledge Base")
# ──────────────────────────────────────────────────────────────

if sidebar["process_button"]:
    sources: list[str] = []

    # Step 1: Load and chunk documents
    with st.spinner("🧠 Loading & chunking documents…"):
        try:
            processor = DocumentProcessor()
            chunks = processor.process(
                pdf_file=sidebar["uploaded_file"],
                youtube_url=sidebar["yt_url"] or None,
            )

            if sidebar["uploaded_file"]:
                sources.append(f"📄 {sidebar['uploaded_file'].name}")
            if sidebar["yt_url"]:
                sources.append(f"🎥 {sidebar['yt_url']}")

        except ValueError as e:
            st.error(f"❌ {e}")
            if "image-based" in str(e) or "scanned" in str(e):
                st.info(
                    "💡 **Tip:** This PDF appears to be scanned (image-only). "
                    "PyPDF cannot extract text from image pages.\n\n"
                    "Try: a text-based PDF, copy-paste the text as a YouTube "
                    "transcript, or use a PDF with selectable text."
                )
            chunks = None
        except RuntimeError as e:
            st.error(f"❌ {e}")
            chunks = None

    # Step 2: Embed & build vector store
    if chunks:
        with st.spinner(
            f"📦 Embedding {len(chunks)} chunks into vector store "
            "(this may take 30–60s on first run)…"
        ):
            manager = VectorStoreManager()
            store = manager.create(chunks)

            st.session_state.vector_store = store
            st.session_state.all_chunks = chunks
            st.session_state.indexed_sources = sources
            st.session_state.chunk_count = len(chunks)
            st.session_state.chat_history = []  # fresh chat for new KB

            st.balloons()
            st.success(
                f"✅ Indexed **{len(chunks)} chunks** from "
                f"**{len(sources)} source(s)**"
            )
            logger.info(
                "Knowledge base synced: %d chunks from %d sources",
                len(chunks), len(sources),
            )


# ──────────────────────────────────────────────────────────────
#  CHAT INTERFACE
# ──────────────────────────────────────────────────────────────

# Render conversation history
render_chat_history()

# Query input
query = st.chat_input("Ask anything about your research…")

if query:
    if st.session_state.vector_store is None:
        st.error("⚠️ Sync your knowledge base first!")
    else:
        # Show user message
        with st.chat_message("user"):
            st.write(query)

        # Generate and stream response
        with st.chat_message("assistant"):
            try:
                # ── Stage 1 & 2: Retrieve ──
                with st.spinner("🔍 Retrieving relevant chunks…"):
                    retriever = HybridRetriever(
                        vector_store=st.session_state.vector_store,
                        all_chunks=st.session_state.all_chunks,
                        top_k=sidebar["top_k"],
                        top_n=sidebar["top_n"],
                        use_reranker=sidebar["use_reranker"],
                        search_type=sidebar["search_type"],
                        bm25_weight=(
                            config.BM25_WEIGHT
                            if sidebar["use_hybrid"]
                            else 0.0
                        ),
                        vector_weight=(
                            config.VECTOR_WEIGHT
                            if sidebar["use_hybrid"]
                            else 1.0
                        ),
                    )
                    docs = retriever.retrieve(query)

                if not docs:
                    st.warning(
                        "No relevant documents found. "
                        "Try rephrasing or uploading more sources."
                    )
                else:
                    # ── Stream answer from LLM ──
                    answer = st.write_stream(
                        stream_answer(
                            query, docs, st.session_state.chat_history
                        )
                    )

                    # ── Show sources ──
                    display_sources(docs)

                    # ── Save to chat history ──
                    st.session_state.chat_history.append(
                        {
                            "question": query,
                            "answer": answer,
                            "sources": build_source_metadata(docs),
                        }
                    )
                    logger.info(
                        "Chat turn saved (history length: %d)",
                        len(st.session_state.chat_history),
                    )

            except Exception as e:
                logger.error("Query failed: %s", e, exc_info=True)
                st.error(f"Error: {e}")