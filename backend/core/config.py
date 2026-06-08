from functools import lru_cache
from typing import List
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "AI Research Assistant"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # ── LLM & Database Configurations ─────────────────────────────────────────
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.1-8b-instant"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    
    CHROMA_PERSIST_DIR: str = "../chroma_data"
    CHROMA_COLLECTION_NAME: str = "research_documents"
    METADATA_DB: str = "../metadata.db"
    UPLOAD_DIR: str = "../uploads"
    
    # ── Vector Search & Context Boundaries ────────────────────────────────────
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64
    MAX_FILE_SIZE_MB: int = 50
    TOP_K_RETRIEVAL: int = 5
    MIN_RELEVANCE_SCORE: float = 0.3
    
    # FIX: Expanded context length to 15000 to prevent severing full document 
    # chunks before they reach the Llama-3 model context window.
    MAX_CONTEXT_LENGTH: int = 15000 
    
    # ── Network & Security ────────────────────────────────────────────────────
    # FIX: Added your port 3000 UI paths alongside localhost networking defaults
    # to guarantee seamless FastAPI connection without CORS errors.
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8501"
    ]
    
    # ── State Management ──────────────────────────────────────────────────────
    SESSION_TTL_HOURS: int = 24
    MAX_HISTORY_TURNS: int = 20

    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()

settings = get_settings()