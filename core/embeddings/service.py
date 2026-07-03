# core/embeddings/service.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from core.config import EmbeddingStrategy

class EmbeddingService(ABC):
    @abstractmethod
    async def embed_text(self, text: str, strategy: EmbeddingStrategy) -> Dict[str, List[float]]:
        """
        Return dict mapping vector_name -> vector.
        For dense: {"dense_text": [0.1, ...]}
        For sparse: {"sparse_text": {index: value}}
        """
        pass

    @abstractmethod
    async def embed_image_path(self, image_path: str, strategy: EmbeddingStrategy) -> Dict[str, List[float]]:
        pass

class DummyEmbeddingService(EmbeddingService):
    """Replace with actual model calls."""
    async def embed_text(self, text: str, strategy: EmbeddingStrategy) -> Dict[str, List[float]]:
        import random
        dim = strategy.dimension
        if strategy.type == "sparse":
            # Sparse vector as dict of indices -> value
            return {strategy.vector_name: {i: random.random() for i in random.sample(range(dim), 10)}}
        else:
            return {strategy.vector_name: [random.random() for _ in range(dim)]}

    async def embed_image_path(self, image_path: str, strategy: EmbeddingStrategy) -> Dict[str, List[float]]:
        # Placeholder
        dim = strategy.dimension
        return {strategy.vector_name: [0.5] * dim}