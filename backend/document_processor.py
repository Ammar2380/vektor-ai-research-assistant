"""
Document processing service.
Handles PDF extraction, chunking, embedding, and ChromaDB storage.
"""

import hashlib
import time
import uuid
from pathlib import Path
from typing import List, Tuple, Dict, Any

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document

from core.config import settings
from core.database import get_collection
from core.logging_config import setup_logging
from services.embedder import EmbeddingService

logger = setup_logging(__name__)


class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )
        self.embedder = EmbeddingService()

    def generate_doc_id(self, filename: str, content_hash: str) -> str:
        """Generate deterministic document ID."""
        return hashlib.sha256(f"{filename}:{content_hash}".encode()).hexdigest()[:16]

    async def process_pdf(
        self,
        file_path: Path,
        filename: str,
        session_id: str | None = None,
        tags: List[str] = [],
    ) -> Dict[str, Any]:
        """
        Full pipeline: load → split → embed → store.
        Returns processing metadata.
        """
        start = time.time()
        logger.info(f"Processing PDF: {filename}")

        # 1. Load PDF
        loader = PyPDFLoader(str(file_path))
        pages = loader.load()
        page_count = len(pages)
        logger.info(f"  Loaded {page_count} pages")

        # 2. Content hash for deduplication
        full_text = " ".join(p.page_content for p in pages)
        content_hash = hashlib.md5(full_text.encode()).hexdigest()
        doc_id = self.generate_doc_id(filename, content_hash)

        # 3. Split into chunks
        chunks: List[Document] = self.text_splitter.split_documents(pages)
        logger.info(f"  Split into {len(chunks)} chunks")

        # 4. Generate embeddings
        texts = [c.page_content for c in chunks]
        embeddings = await self.embedder.embed_batch(texts)
        logger.info(f"  Generated {len(embeddings)} embeddings")

        # 5. Prepare ChromaDB records
        ids, documents, metadatas = [], [], []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"{doc_id}_chunk_{i}"
            ids.append(chunk_id)
            documents.append(chunk.page_content)
            metadatas.append({
                "doc_id": doc_id,
                "filename": filename,
                "chunk_index": i,
                "page_number": chunk.metadata.get("page", 0) + 1,
                "session_id": session_id or "",
                "tags": ",".join(tags),
                "char_count": len(chunk.page_content),
            })

        # 6. Store in ChromaDB
        collection = get_collection()
        collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        elapsed = (time.time() - start) * 1000
        logger.info(f"  Done in {elapsed:.0f}ms")

        return {
            "doc_id": doc_id,
            "filename": filename,
            "page_count": page_count,
            "chunk_count": len(chunks),
            "processing_time_ms": elapsed,
            "content_hash": content_hash,
        }

    async def delete_document(self, doc_id: str) -> int:
        """Remove all chunks for a document."""
        collection = get_collection()
        results = collection.get(where={"doc_id": doc_id})
        if not results["ids"]:
            return 0
        collection.delete(ids=results["ids"])
        logger.info(f"Deleted {len(results['ids'])} chunks for doc {doc_id}")
        return len(results["ids"])

    def list_documents(self) -> List[Dict[str, Any]]:
        """Get unique documents from ChromaDB."""
        collection = get_collection()
        results = collection.get(include=["metadatas"])
        seen = {}
        for meta in results["metadatas"]:
            doc_id = meta["doc_id"]
            if doc_id not in seen:
                seen[doc_id] = meta
        return list(seen.values())
