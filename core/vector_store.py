"""
ChromaDB vector store lifecycle management.

Handles creation, clearing, and document extraction. Each "Sync"
resets the collection to prevent data contamination from prior sessions
without unlinking active SQLite database files from disk while open.
"""

import os
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from config import config
from core.embeddings import get_embedding_model
from utils.logger import setup_logger

logger = setup_logger("nexus.vectorstore")

COLLECTION_NAME = "nexus_research"


class VectorStoreManager:
    """Manages the ChromaDB vector store lifecycle."""

    def __init__(self, persist_dir: str = config.PERSIST_DIR):
        self.persist_dir = persist_dir
        self.embeddings = get_embedding_model()
        self.client = chromadb.PersistentClient(path=self.persist_dir)

    def create(self, chunks: list[Document]) -> Chroma:
        """Build a fresh vector store from document chunks.

        The existing collection is deleted and recreated via Chroma's client API,
        avoiding file-system race conditions or locked SQLite database handles.

        Args:
            chunks: Pre-processed and deduplicated document chunks.

        Returns:
            A Chroma vector store ready for retrieval.
        """
        try:
            self.client.delete_collection(COLLECTION_NAME)
            logger.info("Cleared previous collection '%s'", COLLECTION_NAME)
        except Exception:
            # Collection may not exist yet on first run
            pass

        store = Chroma(
            client=self.client,
            collection_name=COLLECTION_NAME,
            embedding_function=self.embeddings,
            collection_metadata={"hnsw:space": "cosine"},
        )

        if chunks:
            store.add_documents(chunks)

        logger.info(
            "Vector store created: %d chunks at %s",
            len(chunks),
            self.persist_dir,
        )
        return store

    def clear(self) -> None:
        """Clear the collection from the vector store."""
        try:
            self.client.delete_collection(COLLECTION_NAME)
            logger.info("Deleted collection '%s' from %s", COLLECTION_NAME, self.persist_dir)
        except Exception as e:
            logger.warning("Could not clear vector store collection: %s", e)

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

