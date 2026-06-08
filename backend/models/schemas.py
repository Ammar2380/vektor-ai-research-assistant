from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
import uuid

class DocumentMetadata(BaseModel):
    doc_id: str
    filename: str
    file_size: int
    page_count: int
    chunk_count: int
    upload_time: datetime
    session_id: Optional[str] = None
    tags: List[str] = []

class DocumentUploadResponse(BaseModel):
    doc_id: str
    filename: str
    status: str
    chunk_count: int
    page_count: int
    processing_time_ms: float
    message: str

class DocumentListResponse(BaseModel):
    documents: List[DocumentMetadata]
    total: int

class DocumentDeleteResponse(BaseModel):
    doc_id: str
    deleted_chunks: int
    message: str

class Citation(BaseModel):
    doc_id: str
    filename: str
    page_number: Optional[int] = None
    chunk_id: str
    relevance_score: float = Field(..., ge=0.0, le=1.0)
    excerpt: str
    chunk_index: int

class ChatMessage(BaseModel):
    role: str
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = None
    doc_ids: Optional[List[str]] = None
    top_k: int = Field(default=5, ge=1, le=20)
    temperature: float = Field(default=0.1, ge=0.0, le=1.0)

class ChatResponse(BaseModel):
    answer: str
    session_id: str
    citations: List[Citation]
    confidence_score: float
    confidence_label: str
    retrieved_chunks: int
    processing_time_ms: float
    model_used: str
    tokens_used: Optional[int] = None

class SessionCreate(BaseModel):
    name: Optional[str] = None
    doc_ids: Optional[List[str]] = None

class SessionResponse(BaseModel):
    session_id: str
    name: str
    created_at: datetime
    last_active: datetime
    message_count: int
    doc_ids: List[str]

class SessionHistory(BaseModel):
    session_id: str
    messages: List[ChatMessage]
    total_messages: int

class EvaluationRequest(BaseModel):
    question: str
    ground_truth: Optional[str] = None
    doc_ids: Optional[List[str]] = None
    top_k: int = Field(default=5, ge=1, le=20)

class RetrievalMetrics(BaseModel):
    mrr: float
    precision_at_k: float
    recall_at_k: float
    ndcg: float
    avg_relevance_score: float
    retrieved_count: int
    latency_ms: float

class EvaluationResponse(BaseModel):
    question: str
    answer: str
    citations: List[Citation]
    retrieval_metrics: RetrievalMetrics
    answer_confidence: float

class HealthResponse(BaseModel):
    status: str
    version: str
    services: Dict[str, Any]
    timestamp: datetime
