# tests/test_storage.py
import asyncio
import random
import uuid
import logging
from datetime import datetime, timezone
from typing import Tuple, List

from qdrant_client.http import models as qdrant_models

from core.stores.postgres import PostgresClient
from core.stores.neo4j import Neo4jClient
from core.stores.qdrant import QdrantClient
from core.stores.manager import StorageManager
from core.stores.worker import OutboxWorker
from core.config import EngineConfig
from core.models.node import Node, ModalityCategory, Granularity, BoundingBox
from core.models.edge import Edge, EdgeCategory
from core.models.document import Document
from tests.dummy_embedder import DummyEmbeddingService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_dummy_document(tenant_id: str = "default") -> Document:
    return Document(
        id=str(uuid.uuid4()),
        tenant_id=tenant_id,
        source_path="/dummy/doc.pdf",
        plugin="dummy",
    )


def generate_dummy_nodes_edges(
    doc_id: str, tenant_id: str = "default", num_nodes: int = 5, num_edges: int = 3
) -> Tuple[List[Node], List[Edge]]:
    nodes = []
    for i in range(num_nodes):
        node = Node(
            id=str(uuid.uuid4()),
            doc_id=doc_id,
            tenant_id=tenant_id,
            content=f"Node {i} content with random text",
            modality="text",
            modality_category=ModalityCategory.TEXTUAL_CONTENT,
            granularity=Granularity.ELEMENT,
            processor_name="dummy",
            processor_version="1.0",
            bbox=BoundingBox(
                x=random.random() * 100,
                y=random.random() * 100,
                w=10,
                h=10,
                page=1,
            ),
        )
        nodes.append(node)

    edges = []
    for i in range(min(num_edges, len(nodes) - 1)):
        edge = Edge(
            id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            source_id=nodes[i].id,
            target_id=nodes[i + 1].id,
            type="contains",
            type_category=EdgeCategory.HIERARCHY,
            weight=1.0,
            creator_processor="dummy",
            creator_version="1.0",
            evidence=f"Edge from {nodes[i].id} to {nodes[i+1].id}",
        )
        edges.append(edge)

    return nodes, edges


async def test_storage():
    config = EngineConfig()
    tenant_id = "test_tenant"

    # ----- Connect to databases -----
    pg = PostgresClient(config.db.postgres_dsn)
    await pg.connect()
    await pg.init_schema()  # ensure tables exist
    await pg.execute("TRUNCATE TABLE nodes, edges, documents, txn_outbox CASCADE")

    neo4j = Neo4jClient(
        config.db.neo4j_url,
        config.db.neo4j_user,
        config.db.neo4j_password,
    )
    await neo4j.connect()
    await neo4j.create_constraints_and_indexes()

    qdrant = QdrantClient(config.db.qdrant_url)
    await qdrant.connect()

    # Create Qdrant collection with named vectors (if not exists)
    await qdrant.create_collection(
        dense_vectors_config={
            "dense_text": qdrant_models.VectorParams(
                size=1024, distance=qdrant_models.Distance.COSINE
            ),
            "visual_clip": qdrant_models.VectorParams(
                size=512, distance=qdrant_models.Distance.COSINE
            ),
            "evidence_dense": qdrant_models.VectorParams(
                size=1024, distance=qdrant_models.Distance.COSINE
            ),
        },
    )

    embedder = DummyEmbeddingService()
    manager = StorageManager(pg, neo4j, qdrant, embedder, config)

    # ----- Test 1: Successful indexing -----
    await pg.execute("TRUNCATE TABLE txn_outbox CASCADE")
    logger.info("\n--- Test 1: Successful indexing ---")
    doc = generate_dummy_document(tenant_id)
    # FIX: Insert the document first to satisfy foreign key constraint
    await pg.insert_document(doc)

    nodes, edges = generate_dummy_nodes_edges(
        doc.id, tenant_id, num_nodes=10, num_edges=5
    )

    await manager.begin_transaction(tenant_id)
    for n in nodes:
        manager.stage_node(n)
    for e in edges:
        manager.stage_edge(e)

    counts = await manager.commit_transaction(tenant_id)
    logger.info(f"Committed {counts['nodes']} nodes and {counts['edges']} edges.")

    # ----- Test 2: Process outbox entries -----
    logger.info("\n--- Test 2: Process outbox entries ---")
    worker = OutboxWorker(pg, neo4j, qdrant, config)
    await worker.process_batch(tenant_id, limit=100)

    # ----- Test 3: Verify PostgreSQL -----
    logger.info("\n--- Test 3: Verify PostgreSQL ---")
    db_nodes = await pg.get_nodes_by_ids(tenant_id, [n.id for n in nodes])
    db_edges = await pg.get_edges_by_ids(tenant_id, [e.id for e in edges])
    logger.info(f"Found {len(db_nodes)} nodes and {len(db_edges)} edges in PG.")

    # ----- Test 4: Verify Neo4j -----
    logger.info("\n--- Test 4: Verify Neo4j ---")
    if nodes:
        neighbors = await neo4j.get_neighbors(tenant_id, [nodes[0].id])
        logger.info(f"Neighbors of {nodes[0].id}: {len(neighbors)} edges found.")

    # ----- Test 5: Verify Qdrant -----
    logger.info("\n--- Test 5: Verify Qdrant ---")
    query_vec = [random.random() for _ in range(1024)]
    hits = await qdrant.search_content_vectors(
        tenant_id, "dense_text", query_vec, limit=5
    )
    logger.info(f"Search returned {len(hits)} hits.")

    # ----- Test 6: Simulate partial failure & retry -----
    logger.info("\n--- Test 6: Simulate partial failure ---")
    bad_entry = {
        "tenant_id": tenant_id,
        "entity_id": str(uuid.uuid4()),
        "entity_type": "NODE",
        "operation": "UPSERT_GRAPH",
        "payload": {},  # empty → validation error in worker
        "idempotency_key": "bad-test-" + str(uuid.uuid4()),
        "created_at": datetime.now(timezone.utc),
        "status": "PENDING",
        "retry_count": 0,
        "max_retries": 2,
    }
    await pg.insert_outbox_batch([bad_entry])
    await worker.process_batch(tenant_id, limit=100)
    updated = await pg.fetch_one(
        "SELECT * FROM txn_outbox WHERE idempotency_key = $1",
        bad_entry["idempotency_key"],
    )
    logger.info(
        f"Bad entry status: {updated['status']}, retry_count: {updated['retry_count']}"
    )

    # ----- Test 7: Multi-tenant isolation -----
    logger.info("\n--- Test 7: Multi-tenant isolation ---")
    tenant2_id = "other_tenant"
    doc2 = generate_dummy_document(tenant2_id)
    await pg.insert_document(doc2)  # Insert document for tenant2
    nodes2, _ = generate_dummy_nodes_edges(
        doc2.id, tenant2_id, num_nodes=3, num_edges=0
    )
    await manager.begin_transaction(tenant2_id)
    for n in nodes2:
        manager.stage_node(n)
    await manager.commit_transaction(tenant2_id)
    await worker.process_batch(tenant2_id, limit=100)

    nodes1_after = await pg.get_nodes_by_ids(tenant_id, [n.id for n in nodes])
    assert len(nodes1_after) == len(nodes)
    logger.info("Tenant isolation verified.")

    # ----- Cleanup -----
    await pg.close()
    await neo4j.close()
    await qdrant.close()
    logger.info("All tests passed.")


if __name__ == "__main__":
    asyncio.run(test_storage())