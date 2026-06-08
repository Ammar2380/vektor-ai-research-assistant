import asyncio
from typing import List
from sentence_transformers import SentenceTransformer
from core.config import settings
from core.logging_config import setup_logging

logger = setup_logging(__name__)

class EmbeddingService:
    _model = None

    def _get_model(self):
        if self._model is None:
            logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
            EmbeddingService._model = SentenceTransformer(settings.EMBEDDING_MODEL)
            logger.info("Embedding model loaded")
        return self._model

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._embed_sync, texts)

    def _embed_sync(self, texts):
        model = self._get_model()
        embeddings = model.encode(
            texts, batch_size=16,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return embeddings.tolist()

    async def embed_query(self, text: str) -> List[float]:
        results = await self.embed_batch([text])
        return results[0]
