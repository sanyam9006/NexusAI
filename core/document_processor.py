"""
Document loading, preprocessing, and chunking.

Supports PDF and YouTube sources.  Documents are:
  1. Loaded from their source format
  2. Filtered — empty / whitespace-only pages are dropped
  3. Enriched with metadata (source name, type, page number)
  4. Split into overlapping chunks
  5. Deduplicated by chunk content hash

Pipeline order is important:
  load → filter empties → chunk → deduplicate chunks
  (NOT: load → deduplicate pages → chunk)
"""

import hashlib
import os
import tempfile
from typing import Optional

from langchain_community.document_loaders import PyPDFLoader, YoutubeLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import config
from utils.logger import setup_logger

logger = setup_logger("nexus.documents")


class DocumentProcessor:
    """End-to-end pipeline: load → filter → chunk → deduplicate."""

    def __init__(
        self,
        chunk_size: int = config.CHUNK_SIZE,
        chunk_overlap: int = config.CHUNK_OVERLAP,
    ):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", ", ", " ", ""],
            length_function=len,
        )
        logger.info(
            "DocumentProcessor ready (chunk_size=%d, overlap=%d)",
            chunk_size,
            chunk_overlap,
        )

    # ── Loaders ────────────────────────────────────────────────

    def load_pdf(self, uploaded_file) -> list[Document]:
        """Load a PDF from a Streamlit UploadedFile.

        Uses a secure temporary file that is always cleaned up after
        loading, regardless of success or failure.
        """
        docs: list[Document] = []
        tmp_path: Optional[str] = None

        try:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".pdf"
            ) as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name

            loader = PyPDFLoader(tmp_path)
            raw_docs = loader.load()

            content_hash = hashlib.md5(
                uploaded_file.getbuffer()
            ).hexdigest()[:8]

            for i, doc in enumerate(raw_docs):
                doc.metadata.update(
                    {
                        "source": uploaded_file.name,
                        "type": "pdf",
                        "page": i + 1,
                        "content_hash": content_hash,
                    }
                )
            docs = raw_docs
            logger.info(
                "Loaded %d pages from PDF '%s' (hash=%s)",
                len(docs),
                uploaded_file.name,
                content_hash,
            )

        except Exception as e:
            logger.error("PDF loading failed: %s", e, exc_info=True)
            raise RuntimeError(
                f"Failed to load PDF '{uploaded_file.name}': {e}"
            ) from e

        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

        return docs

    def load_youtube(self, url: str) -> list[Document]:
        """Load a YouTube transcript with video metadata."""
        try:
            loader = YoutubeLoader.from_youtube_url(
                url, add_video_info=True
            )
            docs = loader.load()

            for doc in docs:
                doc.metadata["type"] = "youtube"
                doc.metadata["source"] = doc.metadata.get("title", url)

            logger.info(
                "Loaded %d segments from YouTube: %s", len(docs), url
            )
            return docs

        except Exception as e:
            logger.error("YouTube loading failed: %s", e, exc_info=True)
            raise RuntimeError(
                f"Failed to load YouTube transcript: {e}"
            ) from e

    # ── Filtering ──────────────────────────────────────────────

    @staticmethod
    def _filter_empty(documents: list[Document]) -> list[Document]:
        """Remove pages/segments with no extractable text.

        This commonly affects:
        - Scanned PDFs (image-only pages → empty text)
        - Title/blank pages in structured PDFs
        - YouTube videos with auto-generated captions disabled
        """
        valid = [
            doc
            for doc in documents
            if doc.page_content and doc.page_content.strip()
        ]
        removed = len(documents) - len(valid)
        if removed > 0:
            logger.warning(
                "Filtered out %d empty page(s)/segment(s) from %d total",
                removed,
                len(documents),
            )
        return valid

    # ── Chunking & Deduplication ───────────────────────────────

    def chunk_documents(self, documents: list[Document]) -> list[Document]:
        """Split documents into chunks, then deduplicate by content hash.

        Order matters:
          1. Split first  → smaller, semantically meaningful units
          2. Deduplicate  → remove exact-duplicate *chunks* (not pages)
        """
        # Step 1: Filter out empty pages BEFORE chunking
        documents = self._filter_empty(documents)

        if not documents:
            logger.error(
                "No non-empty documents to chunk. "
                "The PDF may be image-based (scanned). "
                "Try a text-based PDF or a YouTube URL instead."
            )
            return []

        # Step 2: Split into chunks
        chunks = self.splitter.split_documents(documents)
        logger.info("Split into %d raw chunks", len(chunks))

        if not chunks:
            logger.warning(
                "Splitter produced 0 chunks from %d documents. "
                "Document content may be too short.",
                len(documents),
            )
            return []

        # Step 3: Deduplicate chunks (not pages) by content hash
        seen: set[str] = set()
        unique: list[Document] = []

        for chunk in chunks:
            h = hashlib.md5(chunk.page_content.encode()).hexdigest()
            if h not in seen:
                seen.add(h)
                chunk.metadata["chunk_hash"] = h
                unique.append(chunk)

        removed = len(chunks) - len(unique)
        if removed > 0:
            logger.info(
                "Deduplication removed %d repeated chunk(s)", removed
            )

        logger.info(
            "Final: %d unique chunks from %d documents",
            len(unique),
            len(documents),
        )
        return unique

    # ── Full Pipeline ──────────────────────────────────────────

    def process(
        self,
        pdf_file=None,
        youtube_url: Optional[str] = None,
    ) -> list[Document]:
        """Run the full pipeline: load → filter → chunk → deduplicate.

        Args:
            pdf_file:     A Streamlit UploadedFile object (or None).
            youtube_url:  A YouTube video URL string (or None).

        Returns:
            A list of unique, non-empty Document chunks ready for
            embedding and indexing.

        Raises:
            ValueError: If no documents could be extracted or chunked.
        """
        all_docs: list[Document] = []

        if pdf_file:
            all_docs.extend(self.load_pdf(pdf_file))

        if youtube_url:
            all_docs.extend(self.load_youtube(youtube_url))

        if not all_docs:
            raise ValueError(
                "No documents could be loaded from the provided sources. "
                "Please check your PDF or YouTube URL."
            )

        chunks = self.chunk_documents(all_docs)

        if not chunks:
            raise ValueError(
                "No text could be extracted. Your PDF may be image-based "
                "(scanned). Try a text-based PDF or a YouTube URL."
            )

        return chunks
