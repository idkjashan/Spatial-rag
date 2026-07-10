# core/stores/manager.py
import uuid
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from core.models.node import Node
from core.models.edge import Edge
from core.models.document import Document
from core.stores.postgres import PostgresClient
from core.stores.neo4j import Neo4jClient
from core.stores.qdrant import QdrantClient
from core.embeddings.graph_embedder import EmbeddingService
from core.config import EngineConfig

logger = logging.getLogger(__name__)

class StorageManager:
    def __init__(
        self,
        pg: PostgresClient,
        neo4j: Neo4jClient,
        qdrant: QdrantClient,
        embedder: EmbeddingService,
        config: EngineConfig
    ):
        self.pg = pg
        self.neo4j = neo4j
        self.qdrant = qdrant
        self.embedder = embedder
        self.config = config
        self._staged_nodes: List[Node] = []
        self._staged_edges: List[Edge] = []
        self._txn_id: Optional[str] = None

    async def begin_transaction(self, tenant_id: str) -> str:
        self._txn_id = str(uuid.uuid4())
        self._staged_nodes = []
        self._staged_edges = []
        logger.debug(f"Transaction {self._txn_id} started for tenant {tenant_id}")
        return self._txn_id

    def stage_node(self, node: Node) -> None:
        if not self._txn_id:
            raise RuntimeError("Must call begin_transaction first.")
        self._staged_nodes.append(node)

    def stage_edge(self, edge: Edge) -> None:
        if not self._txn_id:
            raise RuntimeError("Must call begin_transaction first.")
        self._staged_edges.append(edge)

    async def commit_transaction(self, tenant_id: str) -> Dict[str, int]:
        if not self._txn_id:
            raise RuntimeError("No active transaction.")

        # 1. Generate embeddings (returns both refs and vectors)
        node_emb_data = await self._embed_nodes(tenant_id, self._staged_nodes)
        edge_emb_data = await self._embed_edges(tenant_id, self._staged_edges)

        # 2. Populate embedding_refs and collect vector_data for outbox
        outbox_entries = []

        for node, emb_data in zip(self._staged_nodes, node_emb_data):
            node.embedding_refs = emb_data["embedding_refs"]
            vectors = emb_data["vectors"]
            # Graph outbox entry
            outbox_entries.append({
                "tenant_id": tenant_id,
                "entity_id": node.id,
                "entity_type": "NODE",
                "operation": "UPSERT_GRAPH",
                "payload": node.model_dump(mode="json"),
                "idempotency_key": f"graph-node-{node.id}",
                "created_at": datetime.now(timezone.utc)
            })
            # Vector outbox entry (with actual vectors)
            if vectors:
                outbox_entries.append({
                    "tenant_id": tenant_id,
                    "entity_id": node.id,
                    "entity_type": "NODE",
                    "operation": "UPSERT_VECTOR",
                    "payload": {"node_id": node.id},
                    "embedding_strategy": "bge-m3",  # simplified; you can store strategy name
                    "vector_data": vectors,  # actual vector lists
                    "idempotency_key": f"vector-node-{node.id}",
                    "created_at": datetime.now(timezone.utc)
                })

        for edge, emb_data in zip(self._staged_edges, edge_emb_data):
            edge.embedding_refs = emb_data["embedding_refs"]
            vectors = emb_data["vectors"]
            outbox_entries.append({
                "tenant_id": tenant_id,
                "entity_id": edge.id,
                "entity_type": "EDGE",
                "operation": "UPSERT_GRAPH",
                "payload": edge.model_dump(mode="json"),
                "idempotency_key": f"graph-edge-{edge.id}",
                "created_at": datetime.now(timezone.utc)
            })
            if vectors:
                outbox_entries.append({
                    "tenant_id": tenant_id,
                    "entity_id": edge.id,
                    "entity_type": "EDGE",
                    "operation": "UPSERT_VECTOR",
                    "payload": {"edge_id": edge.id},
                    "embedding_strategy": "bge-m3",
                    "vector_data": vectors,
                    "idempotency_key": f"vector-edge-{edge.id}",
                    "created_at": datetime.now(timezone.utc)
                })

        # 3. Atomic write to PostgreSQL: nodes, edges, outbox in one transaction
        try:
            async with self.pg.transaction():
                await self.pg.insert_nodes_batch(self._staged_nodes)
                await self.pg.insert_edges_batch(self._staged_edges)
                await self.pg.insert_outbox_batch(outbox_entries)
        except Exception as e:
            logger.error(f"Transaction failed: {e}")
            raise  # rollback happens automatically

        # 4. Clear staging
        counts = {"nodes": len(self._staged_nodes), "edges": len(self._staged_edges)}
        self._staged_nodes = []
        self._staged_edges = []
        self._txn_id = None

        # 5. DO NOT spawn worker – dedicated daemon will pick up outbox entries
        logger.info(f"Transaction committed. Nodes: {counts['nodes']}, Edges: {counts['edges']}")
        return counts

    # ---- Internal embedding helpers ----

    async def _embed_nodes(self, tenant_id: str, nodes: List[Node]) -> List[Dict[str, Any]]:
        """Return list of {embedding_refs: dict, vectors: dict}"""
        results = []
        for node in nodes:
            refs = {}
            vectors = {}
            for strategy in self.config.active_node_strategies:
                strategy_def = next(
                    (s for s in self.config.node_embedding_strategies if s.name == strategy),
                    None
                )
                if not strategy_def:
                    continue
                if node.image_path and strategy_def.vector_name == "visual_clip":
                    vec_dict = await self.embedder.embed_image_path(node.image_path, strategy_def)
                else:
                    text = node.content or node.modality
                    vec_dict = await self.embedder.embed_text(text, strategy_def)
                for vec_name, vec_list in vec_dict.items():
                    refs[vec_name] = str(node.id)  # point_id = str(node.id) (since we use node.id as Qdrant point)
                    vectors[vec_name] = vec_list
            results.append({"embedding_refs": refs, "vectors": vectors})
        return results

    async def _embed_edges(self, tenant_id: str, edges: List[Edge]) -> List[Dict[str, Any]]:
        results = []
        for edge in edges:
            refs = {}
            vectors = {}
            for strategy in self.config.active_edge_strategies:
                strategy_def = next(
                    (s for s in self.config.edge_embedding_strategies if s.name == strategy),
                    None
                )
                if not strategy_def:
                    continue
                text = edge.evidence or f"{edge.type} connection from {edge.source_id} to {edge.target_id}"
                vec_dict = await self.embedder.embed_text(text, strategy_def)
                for vec_name, vec_list in vec_dict.items():
                    refs[vec_name] = str(edge.id)
                    vectors[vec_name] = vec_list
            results.append({"embedding_refs": refs, "vectors": vectors})
        return results

    # ---- Hydration methods (for retrieval) ----
    async def get_nodes(self, tenant_id: str, node_ids: List[str]) -> List[Node]:
        return await self.pg.get_nodes_by_ids(tenant_id, node_ids)

    async def get_edges(self, tenant_id: str, edge_ids: List[str]) -> List[Edge]:
        return await self.pg.get_edges_by_ids(tenant_id, edge_ids)

    async def get_neighbors(self, tenant_id: str, node_ids: List[str], categories: Optional[List[str]] = None) -> List[Dict]:
        return await self.neo4j.get_neighbors(tenant_id, node_ids, categories)