"""
Centralized configuration for Nexus AI Research Hub.

All settings are loaded from environment variables with sensible defaults.
The Config dataclass is frozen (immutable) to prevent accidental mutation.

Performance tiers (set via EMBEDDING_MODEL env var):
  Fast  (CPU): all-MiniLM-L6-v2        ~80 MB,  384-dim, <1s load
  Mid   (CPU): all-mpnet-base-v2        ~420 MB, 768-dim, ~10s load
  Best  (GPU): BAAI/bge-base-en-v1.5   ~438 MB, 768-dim, ~10s load on CPU
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Config:
    """Application configuration loaded from environment variables."""

    # ── API Keys ──
    GROQ_API_KEY: str = field(
        default_factory=lambda: os.getenv("GROQ_API_KEY", "")
    )

    # ── Models ──
    # Default: all-MiniLM-L6-v2 = 80MB, fast on CPU.
    # Upgrade to BAAI/bge-base-en-v1.5 if you have a GPU or don't mind ~10s load.
    EMBEDDING_MODEL: str = field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_MODEL", "all-MiniLM-L6-v2"
        )
    )
    RERANKER_MODEL: str = field(
        default_factory=lambda: os.getenv(
            "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
    )
    LLM_MODEL: str = field(
        default_factory=lambda: os.getenv(
            "LLM_MODEL", "openai/gpt-oss-120b"
        )
    )

    # ── Chunking ──
    CHUNK_SIZE: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_SIZE", "1500"))
    )
    CHUNK_OVERLAP: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "200"))
    )

    # ── Retrieval ──
    RETRIEVER_K: int = field(
        default_factory=lambda: int(os.getenv("RETRIEVER_K", "4"))
    )
    RERANKER_TOP_N: int = field(
        default_factory=lambda: int(os.getenv("RERANKER_TOP_N", "3"))
    )
    BM25_WEIGHT: float = field(
        default_factory=lambda: float(os.getenv("BM25_WEIGHT", "0.3"))
    )
    VECTOR_WEIGHT: float = field(
        default_factory=lambda: float(os.getenv("VECTOR_WEIGHT", "0.7"))
    )
    # Reranker is OFF by default — adds ~1-2s per query on CPU.
    # Enable via sidebar toggle or set RERANKER_ENABLED=true in .env.
    RERANKER_ENABLED: bool = field(
        default_factory=lambda: os.getenv("RERANKER_ENABLED", "false").lower() == "true"
    )

    # ── Storage ──
    PERSIST_DIR: str = field(
        default_factory=lambda: os.getenv("CHROMA_PERSIST_DIR", "./nexus_db")
    )

    # ── Chat ──
    MAX_CHAT_HISTORY: int = field(
        default_factory=lambda: int(os.getenv("MAX_CHAT_HISTORY", "10"))
    )

    def validate(self) -> list[str]:
        """Return a list of validation error messages (empty = valid)."""
        errors = []
        if not self.GROQ_API_KEY or self.GROQ_API_KEY.startswith("your-"):
            errors.append(
                "GROQ_API_KEY is missing or still set to the placeholder value."
            )
        if self.CHUNK_OVERLAP >= self.CHUNK_SIZE:
            errors.append(
                f"CHUNK_OVERLAP ({self.CHUNK_OVERLAP}) must be smaller than "
                f"CHUNK_SIZE ({self.CHUNK_SIZE})."
            )
        if self.RERANKER_TOP_N > self.RETRIEVER_K:
            errors.append(
                f"RERANKER_TOP_N ({self.RERANKER_TOP_N}) should not exceed "
                f"RETRIEVER_K ({self.RETRIEVER_K})."
            )
        return errors


# Singleton instance — import this everywhere
config = Config()
