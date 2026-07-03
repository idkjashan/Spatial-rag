# core/stores/qdrant.py
import logging
from typing import List, Dict, Any, Optional
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from core.config import EmbeddingStrategy

logger = logging.getLogger(__name__)

class QdrantClient:
    def __init__(self, url: str, api_key: Optional[str] = None, timeout: int = 60):
        self.url = url
        self.api_key = api_key
        self.timeout = timeout
        self._client: Optional[AsyncQdrantClient] = None
        self._collection_name = "spatialrag_vectors"

    async def connect(self):
        self._client = AsyncQdrantClient(
            url=self.url,
            api_key=self.api_key,
            timeout=self.timeout
        )
        # Verify connectivity
        await self._client.get_collections()
        logger.info("Qdrant connection established.")

    async def close(self):
        if self._client:
            await self._client.close()
            logger.info("Qdrant connection closed.")

    async def create_collection(
        self,
        dense_vectors_config: Optional[Dict[str, models.VectorParams]] = None,
        sparse_vectors_config: Optional[Dict[str, models.SparseVectorParams]] = None,
    ) -> None:
        """
        Create the main collection with named vectors.
        Dense vectors go to dense_vectors_config, sparse to sparse_vectors_config.
        """
        if dense_vectors_config is None:
            dense_vectors_config = {
                "dense_text": models.VectorParams(size=1024, distance=models.Distance.COSINE),
                "visual_clip": models.VectorParams(size=512, distance=models.Distance.COSINE),
                "evidence_dense": models.VectorParams(size=1024, distance=models.Distance.COSINE),
            }
        if sparse_vectors_config is None:
            sparse_vectors_config = {
                "sparse_text": models.SparseVectorParams(),
            }

        # Check if collection already exists
        try:
            await self._client.get_collection(self._collection_name)
            logger.info(f"Collection {self._collection_name} already exists. Skipping creation.")
            return
        except Exception as e:
            if "404" not in str(e) and "Not found" not in str(e):
                raise
            logger.info(f"Collection {self._collection_name} not found. Creating...")

        # Create collection with both dense and sparse configs
        await self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config=dense_vectors_config,
            sparse_vectors_config=sparse_vectors_config,
        )

        # Create payload indexes
        index_fields = [
            ("tenant_id", models.PayloadSchemaType.KEYWORD),
            ("entity_id", models.PayloadSchemaType.KEYWORD),
            ("entity_type", models.PayloadSchemaType.KEYWORD),
            ("modality_category", models.PayloadSchemaType.KEYWORD),
        ]
        for field_name, field_type in index_fields:
            try:
                await self._client.create_payload_index(
                    collection_name=self._collection_name,
                    field_name=field_name,
                    field_type=field_type
                )
            except Exception as e:
                if "already exists" not in str(e).lower():
                    raise
                logger.debug(f"Index on {field_name} already exists.")

        logger.info(f"Qdrant collection {self._collection_name} created/verified.")

    # ---- Batch Upsert with Named Vectors ----

    async def upsert_node_vectors(
        self,
        tenant_id: str,
        node_id: str,
        vectors: Dict[str, List[float]]  # {"dense_text": [...], "visual_clip": [...]}
    ) -> None:
        """
        Upsert one node's vectors. We use the node_id as the point_id.
        Payload stores: tenant_id, entity_id, entity_type, modality_category, subgraph_id.
        """
        point = models.PointStruct(
            id=node_id,
            vector=vectors,  # named vectors dict
            payload={
                "tenant_id": tenant_id,
                "entity_id": node_id,
                "entity_type": "node",
                # You can add more payload fields from the Node if needed
            }
        )
        await self._client.upsert(
            collection_name=self._collection_name,
            points=[point]
        )

    async def upsert_edge_vectors(
        self,
        tenant_id: str,
        edge_id: str,
        vectors: Dict[str, List[float]]  # {"evidence_dense": [...]}
    ) -> None:
        point = models.PointStruct(
            id=edge_id,
            vector=vectors,
            payload={
                "tenant_id": tenant_id,
                "entity_id": edge_id,
                "entity_type": "edge",
            }
        )
        await self._client.upsert(
            collection_name=self._collection_name,
            points=[point]
        )

    # core/stores/qdrant.py (updated methods)

    async def upsert_batch_nodes(self, tenant_id: str, node_data: List[Dict[str, Any]]) -> None:
        if not node_data:
            return
        points = []
        for d in node_data:
            vector_dict = {}
            for name, vec in d["vectors"].items():
                if isinstance(vec, dict) and "indices" in vec and "values" in vec:
                    vector_dict[name] = models.SparseVector(indices=vec["indices"], values=vec["values"])
                else:
                    vector_dict[name] = vec
            points.append(
                models.PointStruct(
                    id=str(d["node_id"]),
                    vector=vector_dict,
                    payload={
                        "tenant_id": tenant_id,
                        "entity_id": str(d["node_id"]),
                        "entity_type": "node",
                        **(d.get("payload", {}))
                    }
                )
            )
        await self._client.upsert(collection_name=self._collection_name, points=points)

    async def upsert_batch_edges(self, tenant_id: str, edge_data: List[Dict[str, Any]]) -> None:
        if not edge_data:
            return
        points = []
        for d in edge_data:
            vector_dict = {}
            for name, vec in d["vectors"].items():
                if isinstance(vec, dict) and "indices" in vec and "values" in vec:
                    vector_dict[name] = models.SparseVector(indices=vec["indices"], values=vec["values"])
                else:
                    vector_dict[name] = vec
            points.append(
                models.PointStruct(
                    id=str(d["edge_id"]),
                    vector=vector_dict,
                    payload={
                        "tenant_id": tenant_id,
                        "entity_id": str(d["edge_id"]),
                        "entity_type": "edge",
                        **(d.get("payload", {}))
                    }
                )
            )
        await self._client.upsert(collection_name=self._collection_name, points=points)

    # ---- Search Methods (Phase 1 Ready) ----

    async def search_content_vectors(
        self,
        tenant_id: str,
        vector_name: str,
        query_vector: List[float],
        limit: int = 10,
        filter_modalities: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Search the content vectors (dense_text, sparse_text, visual_clip).
        Returns list of points with scores.
        """
        q_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="tenant_id",
                    match=models.MatchValue(value=tenant_id)
                ),
                models.FieldCondition(
                    key="entity_type",
                    match=models.MatchValue(value="node")
                )
            ]
        )
        if filter_modalities:
            q_filter.must.append(
                models.FieldCondition(
                    key="modality_category",
                    match=models.MatchAny(any=filter_modalities)
                )
            )

        result = await self._client.search(
            collection_name=self._collection_name,
            query_vector=(vector_name, query_vector),
            query_filter=q_filter,
            limit=limit,
            with_payload=True
        )
        return [hit.model_dump() for hit in result]

    async def search_evidence_vectors(
        self,
        tenant_id: str,
        query_vector: List[float],
        edge_ids: Optional[List[str]] = None,
        limit: int = 10
    ) -> List[Dict]:
        """Search the evidence vectors of edges, optionally filtered by edge_ids."""
        q_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="tenant_id",
                    match=models.MatchValue(value=tenant_id)
                ),
                models.FieldCondition(
                    key="entity_type",
                    match=models.MatchValue(value="edge")
                )
            ]
        )
        if edge_ids:
            q_filter.must.append(
                models.FieldCondition(
                    key="entity_id",
                    match=models.MatchAny(any=edge_ids)
                )
            )

        result = await self._client.search(
            collection_name=self._collection_name,
            query_vector=("evidence_dense", query_vector),
            query_filter=q_filter,
            limit=limit,
            with_payload=True
        )
        return [hit.model_dump() for hit in result]