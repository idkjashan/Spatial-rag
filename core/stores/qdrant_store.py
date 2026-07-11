import logging
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from typing import List, Any

from core.config import DatabaseConfig

logger = logging.getLogger(__name__)

class QdrantStore:
    def __init__(self, config: DatabaseConfig):
        try:
            self.client = QdrantClient(
                url=config.qdrant_url, 
                api_key=config.qdrant_api_key
            )
            logger.info("Qdrant connection verified successfully.")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise

    def _ensure_collection_exists(self, collection_name: str, vector_size: int):
        """
        Lazily creates or recreates a collection to match the exact dimension of the vectors provided.
        """
        collections = self.client.get_collections().collections
        names = [c.name for c in collections]
        
        if collection_name in names:
            # Fetch existing collection config to check dimensions
            col_info = self.client.get_collection(collection_name)
            existing_dim = col_info.config.params.vectors.size
            
            if existing_dim != vector_size:
                logger.warning(
                    f"Dimension mismatch for '{collection_name}' (Existing: {existing_dim}, New: {vector_size}). "
                    f"Dropping and recreating collection..."
                )
                self.client.delete_collection(collection_name)
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
                )
                logger.info(f"Collection '{collection_name}' recreated with dimension {vector_size}.")
        else:
            logger.info(f"Creating Qdrant collection '{collection_name}' with dynamic dimension {vector_size}...")
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            logger.info(f"Collection '{collection_name}' created.")

    def reset(self):
        collections = self.client.get_collections().collections
        names = [c.name for c in collections]
        if "nodes" in names: self.client.delete_collection("nodes")
        if "edges" in names: self.client.delete_collection("edges")
        logger.info("Qdrant collections cleared.")

    def upsert_vectors(self, collection_name: str, items: List[Any], model_name: str = "nomic-embed-text"):
        if not items: 
            logger.info(f"No items provided for Qdrant upsert to '{collection_name}'.")
            return
        
        points = []
        skipped_count = 0
        
        for item in items:
            meta_dict = getattr(item, 'node_meta', None) or getattr(item, 'edge_meta', None)
            if not meta_dict:
                skipped_count += 1
                continue
                
            vector = meta_dict.get("__embedding_vector__")
            if not vector:
                skipped_count += 1
                continue
                
            point_id = str(item.id)
            
            # Build Rich Payload for Hybrid Search & Instant Hydration
            payload = {
                "doc_id": str(getattr(item, 'doc_id', '')),
                "tenant_id": getattr(item, 'tenant_id', 'default')
            }

            # Node-specific payload fields
            if hasattr(item, 'modality'):
                payload["modality"] = item.modality
                payload["modality_category"] = item.modality_category.value if item.modality_category else None
                payload["granularity"] = item.granularity.value if hasattr(item, 'granularity') and item.granularity else None
                payload["subgraph_role"] = getattr(item, 'subgraph_role', None)
                payload["subgraph_id"] = str(item.subgraph_id) if getattr(item, 'subgraph_id', None) else None
                payload["sequence_index"] = getattr(item, 'sequence_index', None)
                payload["content"] = getattr(item, 'content', "")[:500]

            # Edge-specific payload fields
            elif hasattr(item, 'evidence'):
                payload["type"] = getattr(item, 'type', None)
                payload["type_category"] = item.type_category.value if hasattr(item, 'type_category') and item.type_category else None
                payload["content"] = (getattr(item, 'evidence', "") or "")[:500]
                payload["source_id"] = str(getattr(item, 'source_id', ''))
                payload["target_id"] = str(getattr(item, 'target_id', ''))
                payload["weight"] = getattr(item, 'weight', 1.0)
                payload["confidence"] = getattr(item, 'confidence', 1.0)
                
            points.append(PointStruct(
                id=point_id,
                vector=vector,
                payload=payload
            ))
            
            # Update mapping and clean up object to prevent OOM in Postgres JSONB
            item.embedding_refs[model_name] = point_id
            
            if "__embedding_vector__" in meta_dict:
                del meta_dict["__embedding_vector__"]

        logger.info(f"Prepared {len(points)} points for '{collection_name}' (Skipped {skipped_count} items without vectors).")
        
        if points:
            try:
                # DYNAMIC DIMENSION: Measure the actual vector length from the first point
                actual_dim = len(points[0].vector)
                self._ensure_collection_exists(collection_name, actual_dim)
                
                self.client.upsert(
                    collection_name=collection_name,
                    points=points,
                    wait=True
                )
                logger.info(f"✅ Upserted {len(points)} vectors to Qdrant '{collection_name}' using model '{model_name}'.")
            except Exception as e:
                logger.error(f"❌ Qdrant upsert failed for '{collection_name}': {e}", exc_info=True)
                raise
        else:
            logger.warning(f"⚠️ No valid vectors found to upsert for '{collection_name}'.")

    def close(self):
        if self.client:
            self.client.close()
            logger.info("Qdrant client closed.")