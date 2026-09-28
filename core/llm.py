"""
LLM client management.

Connects to Groq's inference API for fast (~800 tok/s) generation
with Llama 3.3 70B.  The client is cached and supports streaming.
"""

import os

import streamlit as st
from langchain_groq import ChatGroq

from config import config
from utils.logger import setup_logger

logger = setup_logger("nexus.llm")


@st.cache_resource(show_spinner="🔄 Connecting to LLM…")
def get_llm(
    model_name: str = config.LLM_MODEL,
    temperature: float = 0,
    streaming: bool = True,
) -> ChatGroq:
    """Load and cache the Groq LLM client.

    Args:
        model_name: Groq-hosted model identifier.
        temperature: Sampling temperature (0 = deterministic).
        streaming: Whether to enable token-by-token streaming.

    Returns:
        A LangChain ChatGroq instance ready for invoke() or stream().
    """
    os.environ["GROQ_API_KEY"] = config.GROQ_API_KEY
    logger.info(
        "Initialising LLM: %s (temp=%.1f, streaming=%s)",
        model_name, temperature, streaming,
    )
    return ChatGroq(
        model_name=model_name,
        temperature=temperature,
        streaming=streaming,
    )
