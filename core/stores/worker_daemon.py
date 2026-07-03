# core/stores/worker_daemon.py
import asyncio
import logging
from core.stores.postgres import PostgresClient
from core.stores.neo4j import Neo4jClient
from core.stores.qdrant import QdrantClient
from core.stores.worker import OutboxWorker
from core.config import EngineConfig

logger = logging.getLogger(__name__)

async def run_worker(tenant_id: str = "default", poll_interval: int = 5):
    config = EngineConfig()
    pg = PostgresClient(config.db.postgres_dsn)
    await pg.connect()
    neo4j = Neo4jClient(config.db.neo4j_url, config.db.neo4j_user, config.db.neo4j_password)
    await neo4j.connect()
    qdrant = QdrantClient(config.db.qdrant_url)
    await qdrant.connect()

    worker = OutboxWorker(pg, neo4j, qdrant, config)
    logger.info(f"Outbox worker started for tenant {tenant_id}, polling every {poll_interval}s")

    while True:
        try:
            await worker.process_batch(tenant_id, limit=100)
        except Exception as e:
            logger.error(f"Worker loop error: {e}")
        await asyncio.sleep(poll_interval)

# Optional: if you want to run multiple tenants, you can start a task per tenant.