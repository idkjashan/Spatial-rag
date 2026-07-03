import logging
from typing import List, Dict, Any, Optional
from neo4j import AsyncGraphDatabase, AsyncDriver
from core.models.node import Node
from core.models.edge import Edge

logger = logging.getLogger(__name__)


class Neo4jClient:
    def __init__(self, url: str, user: str, password: str, max_pool_size: int = 50):
        self.url = url
        self.user = user
        self.password = password
        self.max_pool_size = max_pool_size
        self._driver: Optional[AsyncDriver] = None

    async def connect(self):
        self._driver = AsyncGraphDatabase.driver(
            self.url,
            auth=(self.user, self.password),
            max_connection_pool_size=self.max_pool_size
        )
        async with self._driver.session() as session:
            await session.run("RETURN 1")
        logger.info("Neo4j connection established.")

    async def close(self):
        if self._driver:
            await self._driver.close()
            logger.info("Neo4j connection closed.")

    async def create_constraints_and_indexes(self):
        queries = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Entity) REQUIRE (n.id, n.tenant_id) IS UNIQUE",
            "CREATE INDEX IF NOT EXISTS FOR (n:Entity) ON (n.tenant_id)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Entity) ON (n.category)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Entity) ON (n.subgraph_id)",
        ]
        async with self._driver.session() as session:
            for q in queries:
                await session.run(q)
        logger.info("Neo4j constraints and indexes created.")

    async def upsert_nodes_and_edges(
        self,
        tenant_id: str,
        nodes: List[Node],
        edges: List[Edge],
        chunk_size: int = 1000
    ) -> None:
        """
        Bulk upsert nodes and relationships separately.
        Nodes are merged first, then edges are created using those nodes.
        """
        # ---- Merge nodes ----
        if nodes:
            node_query = """
                UNWIND $nodes AS nodeData
                MERGE (n:Entity {id: nodeData.id, tenant_id: nodeData.tenant_id})
                SET n.category = nodeData.category,
                    n.modality = nodeData.modality,
                    n.subgraph_id = nodeData.subgraph_id,
                    n.granularity = nodeData.granularity
            """
            for i in range(0, len(nodes), chunk_size):
                chunk = nodes[i:i+chunk_size]
                node_params = [
                    {
                        "id": n.id,
                        "tenant_id": n.tenant_id,
                        "category": n.modality_category.value if hasattr(n.modality_category, "value") else str(n.modality_category),
                        "modality": n.modality,
                        "subgraph_id": n.subgraph_id,
                        "granularity": n.granularity.value if hasattr(n.granularity, "value") else str(n.granularity),
                    }
                    for n in chunk
                ]
                async with self._driver.session() as session:
                    await session.run(node_query, nodes=node_params)
                logger.debug(f"Merged {len(chunk)} nodes.")

        # ---- Merge edges ----
        if edges:
            edge_query = """
                UNWIND $edges AS edge
                MATCH (src:Entity {id: edge.source_id, tenant_id: edge.tenant_id})
                MATCH (tgt:Entity {id: edge.target_id, tenant_id: edge.tenant_id})
                MERGE (src)-[r:CONNECTS {edge_id: edge.edge_id}]->(tgt)
                SET r.structural_class = edge.structural_class,
                    r.weight = edge.weight
            """
            for i in range(0, len(edges), chunk_size):
                chunk = edges[i:i+chunk_size]
                edge_params = [
                    {
                        "source_id": e.source_id,
                        "target_id": e.target_id,
                        "edge_id": e.id,
                        "structural_class": e.type_category.value if hasattr(e.type_category, "value") else str(e.type_category),
                        "weight": e.weight,
                        "tenant_id": e.tenant_id,
                    }
                    for e in chunk
                ]
                async with self._driver.session() as session:
                    await session.run(edge_query, edges=edge_params)
                logger.debug(f"Merged {len(chunk)} edges.")

        logger.info(f"Upserted {len(nodes)} nodes and {len(edges)} edges to Neo4j.")

    async def get_neighbors(
        self,
        tenant_id: str,
        node_ids: List[str],
        structural_classes: Optional[List[str]] = None,
        max_depth: int = 1
    ) -> List[Dict]:
        if not node_ids:
            return []
        filter_clause = ""
        params = {"tenant_id": tenant_id, "node_ids": node_ids}
        if structural_classes:
            filter_clause = "AND r.structural_class IN $structural_classes"
            params["structural_classes"] = structural_classes

        query = f"""
            MATCH (n:Entity {{tenant_id: $tenant_id}})-[r:CONNECTS]-(m:Entity {{tenant_id: $tenant_id}})
            WHERE n.id IN $node_ids {filter_clause}
            RETURN n.id AS source_id, m.id AS target_id,
                   r.structural_class AS structural_class,
                   r.weight AS weight,
                   r.edge_id AS edge_id
            LIMIT 1000
        """
        async with self._driver.session() as session:
            result = await session.run(query, **params)
            records = await result.data()
        return records

    async def get_subgraph(self, tenant_id: str, subgraph_id: str) -> List[Dict]:
        query = """
            MATCH (n:Entity {tenant_id: $tenant_id, subgraph_id: $subgraph_id})
            OPTIONAL MATCH (n)-[r:CONNECTS]-(m:Entity {tenant_id: $tenant_id, subgraph_id: $subgraph_id})
            RETURN n.id AS node_id, n.category AS category,
                   r.edge_id AS edge_id, r.structural_class AS structural_class,
                   m.id AS connected_to
        """
        async with self._driver.session() as session:
            result = await session.run(query, tenant_id=tenant_id, subgraph_id=subgraph_id)
            return await result.data()