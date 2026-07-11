import logging
from neo4j import GraphDatabase
from typing import List

from core.models.node import Node
from core.models.edge import Edge
from core.config import DatabaseConfig

logger = logging.getLogger(__name__)

class Neo4jStore:
    def __init__(self, config: DatabaseConfig):
        try:
            self.driver = GraphDatabase.driver(
                config.neo4j_url, 
                auth=(config.neo4j_user, config.neo4j_password)
            )
            # Verify connectivity immediately to fail fast
            self.driver.verify_connectivity()
            self._init_constraints()
            logger.info("Neo4j connection verified successfully.")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise

    def _init_constraints(self):
        with self.driver.session() as session:
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Node) REQUIRE n.id IS UNIQUE;")
            # Create indexes for fast traversal filtering during Query Phase
            session.run("CREATE INDEX IF NOT EXISTS FOR (n:Node) ON (n.modality_category);")
            session.run("CREATE INDEX IF NOT EXISTS FOR (n:Node) ON (n.granularity);")
            session.run("CREATE INDEX IF NOT EXISTS FOR (n:Node) ON (n.subgraph_role);")
            session.run("CREATE INDEX IF NOT EXISTS FOR (n:Node) ON (n.tenant_id);")
            session.run("CREATE INDEX IF NOT EXISTS FOR ()-[r:CONNECTS]-() ON (r.type_category);")
        logger.info("Neo4j constraints and indexes initialized.")

    def reset(self):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n;")
        logger.info("Neo4j graph cleared.")

    def save_nodes(self, nodes: List[Node]):
        if not nodes: return
        data = [
            {
                "id": str(n.id),
                "doc_id": str(n.doc_id),
                "tenant_id": getattr(n, 'tenant_id', 'default'),
                "modality": n.modality,
                "modality_category": n.modality_category.value if n.modality_category else None,
                "granularity": n.granularity.value if hasattr(n, 'granularity') and n.granularity else None,
                "subgraph_role": getattr(n, 'subgraph_role', None),
                "sequence_index": getattr(n, 'sequence_index', None),
                "parent_id": str(n.parent_id) if n.parent_id else None,
                "subgraph_id": str(n.subgraph_id) if n.subgraph_id else None
            } for n in nodes
        ]
        
        query = """
        UNWIND $batch AS row
        MERGE (n:Node {id: row.id})
        SET n.doc_id = row.doc_id, 
            n.tenant_id = row.tenant_id,
            n.modality = row.modality, 
            n.modality_category = row.modality_category,
            n.granularity = row.granularity,
            n.subgraph_role = row.subgraph_role,
            n.sequence_index = row.sequence_index,
            n.parent_id = row.parent_id,
            n.subgraph_id = row.subgraph_id
        """
        with self.driver.session() as session:
            session.run(query, batch=data)
        logger.info(f"Saved {len(nodes)} nodes to Neo4j.")

    def save_edges(self, edges: List[Edge]):
        if not edges: return
        data = []
        for e in edges:
            safe_type = "".join(c if c.isalnum() or c == "_" else "_" for c in e.type.upper())
            data.append({
                "source_id": str(e.source_id),
                "target_id": str(e.target_id),
                "edge_id": str(e.id),
                "tenant_id": getattr(e, 'tenant_id', 'default'),
                "type": safe_type,
                "type_category": e.type_category.value if e.type_category else None,
                "is_bidirectional": e.is_bidirectional,
                "weight": getattr(e, 'weight', 1.0),
                "confidence": getattr(e, 'confidence', 1.0)
            })
            
        query = """
        UNWIND $batch AS row
        MATCH (s:Node {id: row.source_id})
        MATCH (t:Node {id: row.target_id})
        MERGE (s)-[r:CONNECTS {id: row.edge_id}]->(t)
        SET r.type = row.type, 
            r.type_category = row.type_category,
            r.tenant_id = row.tenant_id,
            r.is_bidirectional = row.is_bidirectional,
            r.weight = row.weight,
            r.confidence = row.confidence
        """
        with self.driver.session() as session:
            session.run(query, batch=data)
        logger.info(f"Saved {len(edges)} edges to Neo4j.")

    def close(self):
        if self.driver:
            self.driver.close()
            logger.info("Neo4j driver closed.")