"""
Hybrid retrieval with cross-encoder reranking.

Two-stage retrieval pipeline:

  Stage 1 — Ensemble Retrieval
    Combines BM25 (keyword matching) and vector search (semantic
    similarity) via a weighted Reciprocal Rank Fusion (RRF) merge.
    BM25 catches exact term matches and acronyms; vector search
    captures paraphrases and meaning.

    Note: EnsembleRetriever was removed from the langchain ecosystem
    in recent versions; we implement weighted RRF fusion manually.

  Stage 2 — Cross-Encoder Reranking
    A cross-encoder model (ms-marco-MiniLM) jointly encodes each
    (query, document) pair to produce a more accurate relevance score
    than bi-encoder cosine similarity.  This is too slow to run on
    the full corpus, so it only re-scores the Stage 1 candidates.
"""

import streamlit as st
from collections import defaultdict
from sentence_transformers import CrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from config import config
from utils.logger import setup_logger

logger = setup_logger("nexus.retriever")

# RRF constant — suppresses high rank sensitivity for top documents
_RRF_K = 60


# ──────────────────────────────────────────────────────────────
#  RERANKER  (cached — loaded once)
# ──────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="🔄 Loading reranker model…")
def get_reranker(
    model_name: str = config.RERANKER_MODEL,
) -> CrossEncoder:
    """Load and cache the cross-encoder reranking model.

    Default: cross-encoder/ms-marco-MiniLM-L-6-v2 (22M params).
    Fast enough for CPU inference on 8–15 candidates.
    """
    logger.info("Loading reranker: %s", model_name)
    return CrossEncoder(model_name)


# ──────────────────────────────────────────────────────────────
#  WEIGHTED RRF FUSION
# ──────────────────────────────────────────────────────────────

def _rrf_fuse(
    bm25_docs: list[Document],
    vector_docs: list[Document],
    bm25_weight: float,
    vector_weight: float,
    top_k: int,
) -> list[Document]:
    """Merge BM25 and vector results using weighted Reciprocal Rank Fusion.

    RRF score for doc d: Σ weight_i / (k + rank_i(d))

    Args:
        bm25_docs:     Ordered list of BM25 results (best first).
        vector_docs:   Ordered list of vector results (best first).
        bm25_weight:   Weight applied to BM25 scores.
        vector_weight: Weight applied to vector scores.
        top_k:         Maximum documents to return.

    Returns:
        Deduplicated, re-scored, sorted document list.
    """
    scores: dict[str, float] = defaultdict(float)
    doc_map: dict[str, Document] = {}

    for rank, doc in enumerate(bm25_docs):
        key = doc.page_content[:200]  # fingerprint by first 200 chars
        scores[key] += bm25_weight / (_RRF_K + rank + 1)
        doc_map[key] = doc

    for rank, doc in enumerate(vector_docs):
        key = doc.page_content[:200]
        scores[key] += vector_weight / (_RRF_K + rank + 1)
        doc_map[key] = doc

    sorted_keys = sorted(scores, key=lambda k: scores[k], reverse=True)
    return [doc_map[k] for k in sorted_keys[:top_k]]


# ──────────────────────────────────────────────────────────────
#  HYBRID RETRIEVER
# ──────────────────────────────────────────────────────────────

class HybridRetriever:
    """Two-stage retriever: hybrid search → cross-encoder reranking.

    Args:
        vector_store: A Chroma vector store with embedded documents.
        all_chunks:   All document chunks (needed for BM25 indexing).
        top_k:        Number of candidates from Stage 1.
        top_n:        Number of documents kept after Stage 2 reranking.
        bm25_weight:  Weight for BM25 in the fusion (0 = disabled).
        vector_weight:Weight for vector search in the fusion.
        use_reranker: Whether to apply cross-encoder reranking.
        search_type:  Vector search strategy ("mmr" or "similarity").
    """

    def __init__(
        self,
        vector_store: Chroma,
        all_chunks: list[Document],
        top_k: int = config.RETRIEVER_K,
        top_n: int = config.RERANKER_TOP_N,
        bm25_weight: float = config.BM25_WEIGHT,
        vector_weight: float = config.VECTOR_WEIGHT,
        use_reranker: bool = True,
        search_type: str = "mmr",
    ):
        self.top_k = top_k
        self.top_n = top_n
        self.use_reranker = use_reranker
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight

        # ── Vector Retriever ──
        search_kwargs: dict = {"k": top_k}
        if search_type == "mmr":
            search_kwargs.update({
                "fetch_k": top_k * 3,
                "lambda_mult": 0.7,
            })

        self.vector_retriever = vector_store.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs,
        )

        # ── BM25 Retriever (optional) ──
        self.bm25_retriever = None
        if bm25_weight > 0 and all_chunks:
            self.bm25_retriever = BM25Retriever.from_documents(
                all_chunks, k=top_k
            )
            logger.info(
                "Hybrid retriever: BM25(%.1f) + Vector(%.1f), k=%d, rerank=%s",
                bm25_weight, vector_weight, top_k, use_reranker,
            )
        else:
            logger.info(
                "Vector-only retriever: k=%d, search=%s, rerank=%s",
                top_k, search_type, use_reranker,
            )

        # ── Cross-Encoder Reranker ──
        self.reranker = get_reranker() if use_reranker else None

    def retrieve(self, query: str) -> list[Document]:
        """Execute the full two-stage retrieval pipeline.

        Args:
            query: The user's natural-language question.

        Returns:
            A ranked list of the most relevant Document objects.
        """
        # ── Stage 1: Hybrid (or Vector-only) Retrieval ──
        vector_docs = self.vector_retriever.invoke(query)

        if self.bm25_retriever:
            bm25_docs = self.bm25_retriever.invoke(query)
            candidates = _rrf_fuse(
                bm25_docs=bm25_docs,
                vector_docs=vector_docs,
                bm25_weight=self.bm25_weight,
                vector_weight=self.vector_weight,
                top_k=self.top_k,
            )
            logger.info(
                "Stage 1 — BM25: %d, Vector: %d, Fused: %d candidates",
                len(bm25_docs), len(vector_docs), len(candidates),
            )
        else:
            candidates = vector_docs
            logger.info(
                "Stage 1 — Vector-only: %d candidates", len(candidates)
            )

        if not candidates:
            return []

        # ── Stage 2: Cross-Encoder Reranking ──
        if self.reranker and self.use_reranker:
            candidates = self._rerank(query, candidates)
            logger.info(
                "Stage 2 — Reranked to top %d documents", len(candidates)
            )

        return candidates

    def _rerank(
        self, query: str, docs: list[Document]
    ) -> list[Document]:
        """Re-score documents using the cross-encoder.

        The cross-encoder processes each (query, document) pair jointly,
        producing more accurate relevance scores than bi-encoder
        cosine similarity.

        Args:
            query: The user's question.
            docs:  Candidate documents from Stage 1.

        Returns:
            The top-N documents sorted by cross-encoder score.
        """
        pairs = [(query, doc.page_content) for doc in docs]
        scores = self.reranker.predict(pairs)

        # Sort by score descending and take top_n
        scored = sorted(
            zip(scores, docs), key=lambda x: x[0], reverse=True
        )

        if scored:
            logger.info(
                "Reranker scores — best: %.4f, worst kept: %.4f",
                scored[0][0],
                scored[min(self.top_n - 1, len(scored) - 1)][0],
            )

        return [doc for _, doc in scored[: self.top_n]]
