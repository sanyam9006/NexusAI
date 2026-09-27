"""
Embedding model management.

Uses BAAI/bge-base-en-v1.5 by default — a 768-dimensional model that
ranks in the top 5 on the MTEB retrieval benchmark, significantly
outperforming the commonly-used all-MiniLM-L6-v2 (384-dim).

The model is cached via @st.cache_resource so it is loaded once and
shared across all Streamlit reruns.
"""

import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings

from config import config
from utils.logger import setup_logger

logger = setup_logger("nexus.embeddings")


@st.cache_resource(show_spinner="🔄 Loading embedding model…")
def get_embedding_model(
    model_name: str = config.EMBEDDING_MODEL,
) -> HuggingFaceEmbeddings:
    """Load and cache the HuggingFace embedding model.

    Args:
        model_name: HuggingFace model identifier.
            Default: BAAI/bge-base-en-v1.5 (~438 MB download on first run).

    Returns:
        A LangChain-compatible embedding model instance.
    """
    logger.info("Loading embedding model: %s", model_name)
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
