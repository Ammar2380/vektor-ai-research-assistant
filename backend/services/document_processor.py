import hashlib, time
from pathlib import Path
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore[import]
from langchain_community.document_loaders import PyPDFLoader
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
        )
        self.embedder = EmbeddingService()

    def generate_doc_id(self, filename, content_hash):
        return hashlib.sha256(f"{filename}:{content_hash}".encode()).hexdigest()[:16]

    async def process_pdf(self, file_path, filename, session_id=None, tags=[]):
        start = time.time()
        logger.info(f"Processing: {filename}")
        loader = PyPDFLoader(str(file_path))
        pages = loader.load()
        full_text = " ".join(p.page_content for p in pages)
        content_hash = hashlib.md5(full_text.encode()).hexdigest()
        doc_id = self.generate_doc_id(filename, content_hash)
        chunks = self.text_splitter.split_documents(pages)
        texts = [c.page_content for c in chunks]
        embeddings = await self.embedder.embed_batch(texts)
        ids, documents, metadatas = [], [], []
        for i, (chunk, _) in enumerate(zip(chunks, embeddings)):
            ids.append(f"{doc_id}_chunk_{i}")
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
        collection = get_collection()
        collection.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
        elapsed = (time.time() - start) * 1000
        return {"doc_id": doc_id, "filename": filename, "page_count": len(pages),
                "chunk_count": len(chunks), "processing_time_ms": elapsed}

    async def delete_document(self, doc_id):
        collection = get_collection()
        results = collection.get(where={"doc_id": doc_id})
        if not results["ids"]: return 0
        collection.delete(ids=results["ids"])
        return len(results["ids"])

    def list_documents(self):
        collection = get_collection()
        results = collection.get(include=["metadatas"])
        seen = {}
        for meta in results["metadatas"]:
            if meta["doc_id"] not in seen:
                seen[meta["doc_id"]] = meta
        return list(seen.values())
