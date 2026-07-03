# tests/dummy_embedder.py
import random
import hashlib
from typing import Dict, List, Any
from core.embeddings.service import EmbeddingService
from core.config import EmbeddingStrategy

class DummyEmbeddingService(EmbeddingService):
    def _deterministic_vector(self, text: str, dim: int) -> List[float]:
        """Generate a deterministic vector from text hash."""
        hash_bytes = hashlib.sha256(text.encode()).digest()
        random.seed(int.from_bytes(hash_bytes[:8], 'big'))
        vec = [random.random() for _ in range(dim)]
        random.seed()  # reset
        return vec

    async def embed_text(self, text: str, strategy: EmbeddingStrategy) -> Dict[str, List[float]]:
        dim = strategy.dimension
        if strategy.type == "sparse":
            indices = sorted(random.sample(range(1000), 5))
            values = [random.random() for _ in range(5)]
            return {strategy.vector_name: {"indices": indices, "values": values}}
        else:
            return {strategy.vector_name: self._deterministic_vector(text, dim)}

    async def embed_image_path(self, image_path: str, strategy: EmbeddingStrategy) -> Dict[str, List[float]]:
        dim = strategy.dimension
        return {strategy.vector_name: self._deterministic_vector(image_path, dim)}