import chromadb
from chromadb.config import Settings as ChromaSettings
from core.config import settings
from core.logging_config import setup_logging

logger = setup_logging(__name__)
_client = None
_collection = None

async def init_chroma():
    global _client, _collection
    _client = chromadb.PersistentClient(
        path=settings.CHROMA_PERSIST_DIR,
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    _collection = _client.get_or_create_collection(
        name=settings.CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info(f"ChromaDB ready")

def get_client():
    if _client is None:
        raise RuntimeError("ChromaDB not initialized")
    return _client

def get_collection():
    if _collection is None:
        raise RuntimeError("ChromaDB collection not initialized")
    return _collection
