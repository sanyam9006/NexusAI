"""
ChromaDB vector store lifecycle management.

Handles creation, clearing, and document extraction.  Each "Sync"
wipes the old store to prevent data contamination from prior sessions.
"""

import os
import shutil

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from config import config
from core.embeddings import get_embedding_model
from utils.logger import setup_logger

logger = setup_logger("nexus.vectorstore")


class VectorStoreManager:
    """Manages the ChromaDB vector store on disk."""

    def __init__(self, persist_dir: str = config.PERSIST_DIR):
        self.persist_dir = persist_dir
        self.embeddings = get_embedding_model()

    def create(self, chunks: list[Document]) -> Chroma:
        """Build a fresh vector store from document chunks.

        Any existing data at ``persist_dir`` is wiped first to ensure
        a clean index without stale documents from prior sessions.

        Args:
            chunks: Pre-processed and deduplicated document chunks.

        Returns:
            A Chroma vector store ready for retrieval.
        """
        # Wipe existing data to prevent contamination
        if os.path.exists(self.persist_dir):
            shutil.rmtree(self.persist_dir)
            logger.info("Cleared stale vector store at %s", self.persist_dir)

        store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_dir,
            collection_metadata={"hnsw:space": "cosine"},
        )

        logger.info(
            "Vector store created: %d chunks at %s",
            len(chunks), self.persist_dir,
        )
        return store

    def clear(self) -> None:
        """Delete the vector store from disk entirely."""
        if os.path.exists(self.persist_dir):
            shutil.rmtree(self.persist_dir)
            logger.info("Deleted vector store at %s", self.persist_dir)

    def get_all_documents(self, store: Chroma) -> list[Document]:
        """Extract all documents from a Chroma store.

        This is used to build the BM25 index for hybrid retrieval,
        since BM25 needs the raw document list (not just embeddings).

        Args:
            store: An existing Chroma vector store.

        Returns:
            A list of LangChain Documents with content and metadata.
        """
        result = store.get(include=["documents", "metadatas"])
        docs = []
        for content, metadata in zip(
            result["documents"], result["metadatas"]
        ):
            docs.append(
                Document(page_content=content, metadata=metadata or {})
            )
        logger.info(
            "Extracted %d documents from vector store for BM25 indexing",
            len(docs),
        )
        return docs
