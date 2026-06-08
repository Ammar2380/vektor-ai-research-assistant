from typing import List, Optional
from core.config import settings
from core.database import get_collection
from core.logging_config import setup_logging
from models.schemas import Citation
from services.embedder import EmbeddingService

logger = setup_logging(__name__)

class RetrieverService:
    def __init__(self):
        self.embedder = EmbeddingService()

    async def retrieve(self, query, top_k=None, doc_ids=None):
        top_k = top_k or settings.TOP_K_RETRIEVAL
        collection = get_collection()
        query_embedding = await self.embedder.embed_query(query)
        where_filter = None
        
        if doc_ids:
            where_filter = {"doc_id": doc_ids[0]} if len(doc_ids) == 1 else {"doc_id": {"$in": doc_ids}}
            
        count = collection.count()
        if count == 0:
            return [], 0.0
            
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, count),
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )
        
        citations = []
        for dist, doc_text, meta in zip(results["distances"][0], results["documents"][0], results["metadatas"][0]):
            score = max(0.0, 1.0 - dist)
            if score < settings.MIN_RELEVANCE_SCORE:
                continue
            
            citations.append(Citation(
                doc_id=meta["doc_id"], 
                filename=meta["filename"],
                page_number=meta.get("page_number"),
                chunk_id=f"{meta['doc_id']}_chunk_{meta['chunk_index']}",
                relevance_score=round(score, 4),
                # FIX: Removed the [:400] truncation to preserve mathematical context 
                excerpt=doc_text.strip(), 
                chunk_index=meta["chunk_index"],
            ))
            
        citations.sort(key=lambda c: c.relevance_score, reverse=True)
        avg = sum(c.relevance_score for c in citations) / len(citations) if citations else 0.0
        return citations, avg

    def build_context(self, citations):
        parts = []
        for i, c in enumerate(citations, 1):
            parts.append(f"[Source {i}: {c.filename}, Page {c.page_number}]\n{c.excerpt}")
        return "\n\n---\n\n".join(parts)[:settings.MAX_CONTEXT_LENGTH]