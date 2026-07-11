import logging
from typing import List

from core.config import EngineConfig
from core.models.node import Node
from core.models.edge import Edge
from core.models.document import Document

from core.stores.postgres_store import PostgresStore
from core.stores.neo4j_store import Neo4jStore
from core.stores.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)

class StorageManager:
    def __init__(self, config: EngineConfig):
        logger.info("Initializing Storage Manager...")
        self.config = config
        self.pg = PostgresStore(config.db)
        self.neo4j = Neo4jStore(config.db)
        # No vector_size passed - Qdrant will auto-detect from the embedder!
        self.qdrant = QdrantStore(config.db)
        
    def reset_database(self):
        """Wipes all data from all three databases. Use with caution!"""
        logger.warning("RESETTING ALL DATABASES...")
        self.qdrant.reset()
        self.neo4j.reset()
        self.pg.reset()
        logger.info("All databases reset successfully.")

    def save(self, document: Document, nodes: List[Node], edges: List[Edge]):
        """Saves the entire document graph to all storage layers."""
        logger.info(f"Starting storage pipeline for Doc {document.id}...")
        
        # Get the model name dynamically from config, fallback to nomic if not set
        model_name = "nomic-embed-text"
        if self.config.node_embedding_strategies:
            model_name = self.config.node_embedding_strategies[0].model_name

        # 1. Qdrant (Extracts vectors, updates embedding_refs, CLEANS meta dicts)
        try:
            self.qdrant.upsert_vectors("nodes", nodes, model_name=model_name)
            self.qdrant.upsert_vectors("edges", edges, model_name=model_name)
        except Exception as e:
            logger.error(f"Qdrant upsert failed: {e}")

        # 2. Postgres (Saves metadata. Meta dicts no longer contain massive vector arrays)
        try:
            self.pg.save_document(document)
            self.pg.save_nodes(nodes)
            self.pg.save_edges(edges)
        except Exception as e:
            logger.error(f"Postgres save failed: {e}")

        # 3. Neo4j (Saves topology skeleton)
        try:
            self.neo4j.save_nodes(nodes)
            self.neo4j.save_edges(edges)
        except Exception as e:
            logger.error(f"Neo4j save failed: {e}")
            
        logger.info("Storage pipeline completed.")

    def close(self):
        """Closes all database connections."""
        self.pg.close()
        self.neo4j.close()
        self.qdrant.close()
        logger.info("All database connections closed.")