from datetime import datetime
from fastapi import APIRouter
from models.schemas import HealthResponse
from core.config import settings
from core.database import get_collection

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
async def health_check():
    services = {}
    try:
        col = get_collection()
        services["chromadb"] = {"status": "ok", "chunks": col.count()}
    except Exception as e:
        services["chromadb"] = {"status": "error", "error": str(e)}
    try:
        from services.embedder import EmbeddingService
        services["embedding"] = {"status": "ok", "model": settings.EMBEDDING_MODEL}
    except Exception as e:
        services["embedding"] = {"status": "error", "error": str(e)}
    overall = "healthy" if all(s.get("status") == "ok" for s in services.values()) else "degraded"
    return HealthResponse(status=overall, version=settings.VERSION, services=services, timestamp=datetime.utcnow())

@router.get("/")
async def root():
    return {"name": settings.APP_NAME, "version": settings.VERSION, "docs": "/docs"}
