"""
Retrieval service — semantic search over ChromaDB.
Computes relevance scores and assembles citations.
"""

import time
from typing import List, Optional, Dict, Any
from core.config import settings
from core.database import get_collection
from core.logging_config import setup_logging
from models.schemas import Citation
from services.embedder import EmbeddingService

logger = setup_logging(__name__)


class RetrieverService:
    def __init__(self):
        self.embedder = EmbeddingService()

    async def retrieve(
        self,
        query: str,
        top_k: int = None,
        doc_ids: Optional[List[str]] = None,
    ) -> tuple[List[Citation], float]:
        """
        Retrieve relevant chunks for a query.
        Returns (citations, avg_relevance_score).
        """
        top_k = top_k or settings.TOP_K_RETRIEVAL
        collection = get_collection()

        # Embed query
        query_embedding = await self.embedder.embed_query(query)

        # Build filter
        where_filter = None
        if doc_ids:
            if len(doc_ids) == 1:
                where_filter = {"doc_id": doc_ids[0]}
            else:
                where_filter = {"doc_id": {"$in": doc_ids}}

        # Query ChromaDB
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, collection.count() or 1),
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        citations = []
        distances = results["distances"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]

        for dist, doc_text, meta in zip(distances, documents, metadatas):
            # ChromaDB cosine distance → similarity score
            score = max(0.0, 1.0 - dist)
            if score < settings.MIN_RELEVANCE_SCORE:
                continue

            citations.append(Citation(
                doc_id=meta["doc_id"],
                filename=meta["filename"],
                page_number=meta.get("page_number"),
                chunk_id=f"{meta['doc_id']}_chunk_{meta['chunk_index']}",
                relevance_score=round(score, 4),
                excerpt=doc_text[:400].strip(),
                chunk_index=meta["chunk_index"],
            ))

        # Sort by relevance
        citations.sort(key=lambda c: c.relevance_score, reverse=True)
        avg_score = (
            sum(c.relevance_score for c in citations) / len(citations)
            if citations else 0.0
        )

        logger.info(
            f"Retrieved {len(citations)} chunks for query "
            f"(avg score: {avg_score:.3f})"
        )
        return citations, avg_score

    def build_context(self, citations: List[Citation]) -> str:
        """Assemble retrieved chunks into a context string."""
        parts = []
        for i, c in enumerate(citations, 1):
            header = f"[Source {i}: {c.filename}, Page {c.page_number}]"
            parts.append(f"{header}\n{c.excerpt}")
        context = "\n\n---\n\n".join(parts)
        return context[: settings.MAX_CONTEXT_LENGTH]
