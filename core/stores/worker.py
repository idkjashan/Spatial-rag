import json
import logging
from typing import List, Dict, Any
from core.stores.postgres import PostgresClient
from core.stores.neo4j import Neo4jClient
from core.stores.qdrant import QdrantClient
from core.config import EngineConfig
from core.models.node import Node
from core.models.edge import Edge

logger = logging.getLogger(__name__)


def _safe_parse_payload(payload: Any) -> Dict:
    """Ensure payload is a dict. If it's a JSON string, parse it."""
    if isinstance(payload, str):
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse payload JSON: {payload[:100]}...")
            raise
    return payload


class OutboxWorker:
    def __init__(self, pg: PostgresClient, neo4j: Neo4jClient, qdrant: QdrantClient, config: EngineConfig):
        self.pg = pg
        self.neo4j = neo4j
        self.qdrant = qdrant
        self.config = config
        self.batch_size = 100

    async def process_batch(self, tenant_id: str, limit: int = 100):
        entries = await self.pg.fetch_pending_outbox(tenant_id, limit=limit)
        if not entries:
            return

        logger.info(f"Processing {len(entries)} outbox entries for tenant {tenant_id}")

        graph_entries = [e for e in entries if e["operation"] == "UPSERT_GRAPH"]
        vector_entries = [e for e in entries if e["operation"] == "UPSERT_VECTOR"]

        if graph_entries:
            await self._process_graph_entries(tenant_id, graph_entries)

        if vector_entries:
            await self._process_vector_entries(tenant_id, vector_entries)

    async def _process_graph_entries(self, tenant_id: str, entries: List[Dict]):
        nodes = []
        edges = []
        entry_ids = [e["id"] for e in entries]

        for entry in entries:
            payload = _safe_parse_payload(entry["payload"])
            entity_type = entry["entity_type"]
            try:
                if entity_type == "NODE":
                    nodes.append(Node(**payload))
                elif entity_type == "EDGE":
                    edges.append(Edge(**payload))
            except Exception as e:
                logger.error(f"Failed to reconstruct {entity_type} from payload: {e}")
                # Mark as FAILED immediately if payload is corrupt
                await self.pg.update_outbox_status([entry["id"]], "FAILED", str(e))
                continue

        if nodes or edges:
            try:
                await self.neo4j.upsert_nodes_and_edges(tenant_id, nodes, edges)
                # Mark only the entries that we actually processed
                await self.pg.update_outbox_status(entry_ids, "DONE")
                logger.info(f"Graph upsert successful for {len(entry_ids)} entries.")
            except Exception as e:
                logger.error(f"Graph upsert failed: {e}")
                for entry_id in entry_ids:
                    retry_count = next(e["retry_count"] for e in entries if e["id"] == entry_id)
                    max_retries = next(e["max_retries"] for e in entries if e["id"] == entry_id)
                    if retry_count >= max_retries:
                        await self.pg.update_outbox_status([entry_id], "FAILED", str(e))
                    else:
                        await self.pg.update_outbox_status([entry_id], "PENDING", str(e), increment_retry=True)

    async def _process_vector_entries(self, tenant_id: str, entries: List[Dict]):
        node_upserts = []
        edge_upserts = []
        entry_ids = [e["id"] for e in entries]

        for entry in entries:
            entity_type = entry["entity_type"]
            entity_id = entry["entity_id"]
            vector_data = entry.get("vector_data")
            # asyncpg returns JSONB as dict, but if we stored as string it might be string, so parse
            if isinstance(vector_data, str):
                try:
                    vector_data = json.loads(vector_data)
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse vector_data for {entity_id}")
                    continue
            if not vector_data:
                logger.warning(f"No vector_data for {entity_id}; skipping.")
                continue

            if entity_type == "NODE":
                node_upserts.append({
                    "node_id": entity_id,
                    "vectors": vector_data,
                    "payload": {"entity_type": "node"}
                })
            elif entity_type == "EDGE":
                edge_upserts.append({
                    "edge_id": entity_id,
                    "vectors": vector_data,
                    "payload": {"entity_type": "edge"}
                })

        try:
            if node_upserts:
                await self.qdrant.upsert_batch_nodes(tenant_id, node_upserts)
            if edge_upserts:
                await self.qdrant.upsert_batch_edges(tenant_id, edge_upserts)
            await self.pg.update_outbox_status(entry_ids, "DONE")
            logger.info(f"Vector upsert successful for {len(entry_ids)} entries.")
        except Exception as e:
            logger.error(f"Vector upsert failed: {e}")
            for entry_id in entry_ids:
                retry_count = next(e["retry_count"] for e in entries if e["id"] == entry_id)
                max_retries = next(e["max_retries"] for e in entries if e["id"] == entry_id)
                if retry_count >= max_retries:
                    await self.pg.update_outbox_status([entry_id], "FAILED", str(e))
                else:
                    await self.pg.update_outbox_status([entry_id], "PENDING", str(e), increment_retry=True)