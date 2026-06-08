from pathlib import Path
from typing import Optional
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from core.config import settings
from models.schemas import DocumentUploadResponse, DocumentListResponse, DocumentDeleteResponse
from services.document_processor import DocumentProcessor
from datetime import datetime

router = APIRouter()
processor = DocumentProcessor()
UPLOAD_DIR = Path(settings.UPLOAD_DIR)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@router.post("/upload", response_model=DocumentUploadResponse, status_code=201)
async def upload_document(file: UploadFile = File(...), session_id: Optional[str] = Form(None), tags: Optional[str] = Form("")):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "Only PDF files supported.")
    content = await file.read()
    if len(content) / (1024*1024) > settings.MAX_FILE_SIZE_MB:
        raise HTTPException(413, f"File exceeds {settings.MAX_FILE_SIZE_MB}MB limit.")
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as f:
        f.write(content)
    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    result = await processor.process_pdf(file_path, file.filename, session_id, tag_list)
    return DocumentUploadResponse(
        doc_id=result["doc_id"], filename=result["filename"], status="processed",
        chunk_count=result["chunk_count"], page_count=result["page_count"],
        processing_time_ms=result["processing_time_ms"],
        message=f"Processed {result['page_count']} pages into {result['chunk_count']} chunks.")

@router.get("/", response_model=DocumentListResponse)
async def list_documents():
    docs = processor.list_documents()
    from models.schemas import DocumentMetadata
    doc_list = [DocumentMetadata(
        doc_id=d["doc_id"], filename=d["filename"], file_size=0,
        page_count=0, chunk_count=0, upload_time=datetime.utcnow(),
        session_id=d.get("session_id") or None,
        tags=d.get("tags","").split(",") if d.get("tags") else [],
    ) for d in docs]
    return DocumentListResponse(documents=doc_list, total=len(doc_list))

@router.delete("/{doc_id}", response_model=DocumentDeleteResponse)
async def delete_document(doc_id: str):
    deleted = await processor.delete_document(doc_id)
    if deleted == 0:
        raise HTTPException(404, f"Document {doc_id} not found.")
    return DocumentDeleteResponse(doc_id=doc_id, deleted_chunks=deleted, message=f"Deleted {deleted} chunks.")

@router.get("/{doc_id}/chunks")
async def get_chunks(doc_id: str, limit: int = 10, offset: int = 0):
    from core.database import get_collection
    collection = get_collection()
    results = collection.get(where={"doc_id": doc_id}, include=["documents","metadatas"], limit=limit, offset=offset)
    if not results["ids"]:
        raise HTTPException(404, "Document not found.")
    return {"doc_id": doc_id, "chunks": [
        {"chunk_id": cid, "text": doc, "metadata": meta}
        for cid, doc, meta in zip(results["ids"], results["documents"], results["metadatas"])
    ]}
