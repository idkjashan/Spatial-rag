import re
import asyncio
import logging
from typing import List, Dict, Tuple, Optional

from openai import AsyncOpenAI, RateLimitError, APIConnectionError, APITimeoutError
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

from core.models.node import Node, ModalityCategory
from core.models.edge import Edge, EdgeCategory
from core.models.document import Document

logger = logging.getLogger(__name__)

class GraphEmbedder:
    """
    Phase 2.5: Graph Embedding Pipeline (Pillar A & C)
    Computes dense vectors for Node content (Pillar A) and Edge evidence (Pillar C).
    Decoupled from storage: returns vector maps to be consumed by the Storage Interface.
    """

    # --- PILLAR A: Node Embedding Rules ---
    EMBEDDABLE_NODE_CATEGORIES = [
        ModalityCategory.TEXTUAL_CONTENT,
        ModalityCategory.TABLE_CONTAINER,
        ModalityCategory.TABLE_CONTENT,
        ModalityCategory.IMAGE,
        ModalityCategory.EQUATION,
        ModalityCategory.DOCUMENT_STRUCTURE
    ]

    # --- PILLAR C: Edge Embedding Rules ---
    EMBEDDABLE_EDGE_CATEGORIES = [
        EdgeCategory.REFERENCE,
        EdgeCategory.SEMANTIC_RELATION,
        EdgeCategory.PHYSICAL_CONNECTION,
        EdgeCategory.FLOW_DIRECTION,
        EdgeCategory.SIGNAL_FLOW,
        EdgeCategory.CAPTION 
    ]

    MAX_EMBED_CHARS = 8000

    def __init__(
        self,
        model_name: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
        batch_size: int = 64,
        max_concurrent_batches: int = 4
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.semaphore = asyncio.Semaphore(max_concurrent_batches)
        
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=120.0,
            max_retries=0 
        )

    def process(self, document: Document, nodes: List[Node], edges: List[Edge]) -> Tuple[Document, List[Node], List[Edge], Dict[str, List[float]], Dict[str, List[float]]]:
        """
        Synchronous wrapper.
        """
        logger.info(f"Starting Graph Embedding (Pillar A & C) for {len(nodes)} nodes and {len(edges)} edges...")
        try:
            # This will safely run the async loop if none is running
            return asyncio.run(self.aprocess(document, nodes, edges))
        except RuntimeError as e:
            if "asyncio.run() cannot be called from a running event loop" in str(e):
                raise RuntimeError(
                    "Cannot call `process()` from a running async environment (like FastAPI/Jupyter). "
                    "Use `await embedder.aprocess(...)` instead."
                ) from e
            raise

    async def aprocess(self, document: Document, nodes: List[Node], edges: List[Edge]) -> Tuple[Document, List[Node], List[Edge], Dict[str, List[float]], Dict[str, List[float]]]:
        """
        Asynchronous process method. Gathers all embedding tasks concurrently.
        """
        node_vector_map: Dict[str, List[float]] = {}
        edge_vector_map: Dict[str, List[float]] = {}
        
        tasks = []

        # 1. Prepare Node Embedding Tasks (Pillar A)
        nodes_to_embed = [
            n for n in nodes 
            if n.modality_category in self.EMBEDDABLE_NODE_CATEGORIES 
            and n.content.strip()
            and not n.embedding_refs.get(self.model_name)
        ]
        
        for i in range(0, len(nodes_to_embed), self.batch_size):
            batch = nodes_to_embed[i:i + self.batch_size]
            tasks.append(self._process_node_batch(batch, node_vector_map))

        # 2. Prepare Edge Embedding Tasks (Pillar C)
        edges_to_embed = [
            e for e in edges 
            if e.type_category in self.EMBEDDABLE_EDGE_CATEGORIES 
            and e.evidence 
            and e.evidence.strip()
            and not e.embedding_refs.get(self.model_name)
        ]
        
        for i in range(0, len(edges_to_embed), self.batch_size):
            batch = edges_to_embed[i:i + self.batch_size]
            tasks.append(self._process_edge_batch(batch, edge_vector_map))

        if tasks:
            await asyncio.gather(*tasks)

        # 3. Attach vectors temporarily to meta for the storage interface to consume
        for node in nodes:
            if node.id in node_vector_map:
                node.node_meta["__embedding_vector__"] = node_vector_map[node.id]
                
        for edge in edges:
            if edge.id in edge_vector_map:
                edge.edge_meta["__embedding_vector__"] = edge_vector_map[edge.id]

        document.touch()
        logger.info(f"Successfully generated {len(node_vector_map)} node vectors and {len(edge_vector_map)} edge vectors.")
        return document, nodes, edges, node_vector_map, edge_vector_map

    # ==========================================
    # Data Preparation (Sanitization)
    # ==========================================

    def _prepare_node_text(self, node: Node) -> str:
        content = node.content
        
        # Strip LLM/VLM tags but keep the text
        content = content.replace("[CAPTION]:", "Caption:")
        content = content.replace("[SLM_SUMMARY]:", "Summary:")
        content = content.replace("[VLM_SUMMARY]:", "Summary:")
        content = content.replace("[SLM_EXPLANATION]:", "Explanation:")
        
        # Remove markdown table syntax for cleaner semantic meaning
        if node.modality_category == ModalityCategory.TABLE_CONTAINER:
            content = re.sub(r'\|', ' ', content)
            content = re.sub(r'-{3,}', '', content)
            content = re.sub(r'\s+', ' ', content).strip()
            
        if len(content) > self.MAX_EMBED_CHARS:
            content = content[:self.MAX_EMBED_CHARS]
            
        return content.strip()

    def _prepare_edge_text(self, edge: Edge) -> Optional[str]:
        evidence = edge.evidence.strip()
        if not evidence:
            return None
            
        # Skip generic fallback templates generated by PostProcessor
        if evidence.startswith("Relationship '"):
            return None
            
        if len(evidence) > self.MAX_EMBED_CHARS:
            evidence = evidence[:self.MAX_EMBED_CHARS]
            
        return evidence

    # ==========================================
    # API Interaction & Batching
    # ==========================================

    @retry(
        retry=retry_if_exception_type((RateLimitError, APIConnectionError, APITimeoutError)),
        stop=stop_after_attempt(5),
        wait=wait_fixed(2),
        before_sleep=lambda retry_state: logger.warning(f"Embedding API limit/network error. Retrying...")
    )
    async def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        async with self.semaphore:
            response = await self.client.embeddings.create(
                model=self.model_name,
                input=texts
            )
            sorted_data = sorted(response.data, key=lambda x: x.index)
            return [d.embedding for d in sorted_data]

    async def _process_node_batch(self, batch: List[Node], vector_map: Dict[str, List[float]]):
        texts_to_embed = []
        valid_nodes = []

        for node in batch:
            clean_text = self._prepare_node_text(node)
            if clean_text:
                texts_to_embed.append(clean_text)
                valid_nodes.append(node)

        if not texts_to_embed:
            return

        try:
            vectors = await self._get_embeddings(texts_to_embed)
            for i, node in enumerate(valid_nodes):
                if i < len(vectors):
                    vector_map[node.id] = vectors[i]
            logger.info(f"  [Nodes] Embedded batch of {len(valid_nodes)} items.")
        except Exception as e:
            logger.error(f"Failed to process node embedding batch: {e}")

    async def _process_edge_batch(self, batch: List[Edge], vector_map: Dict[str, List[float]]):
        texts_to_embed = []
        valid_edges = []

        for edge in batch:
            clean_text = self._prepare_edge_text(edge)
            if clean_text:
                texts_to_embed.append(clean_text)
                valid_edges.append(edge)

        if not texts_to_embed:
            return

        try:
            vectors = await self._get_embeddings(texts_to_embed)
            for i, edge in enumerate(valid_edges):
                if i < len(vectors):
                    # FIX: Edges inherit from SpatialRAGBase, so they have a native UUID 'id'
                    vector_map[edge.id] = vectors[i]
            logger.info(f"  [Edges] Embedded batch of {len(valid_edges)} items.")
        except Exception as e:
            logger.error(f"Failed to process edge embedding batch: {e}")