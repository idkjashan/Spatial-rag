1. List of All Storage Infrastructure Components (for Archival)
We implemented the following classes, methods, and modules in the core/stores/ directory, along with supporting models and configuration.

Module	Key Classes / Functions	Key Methods	Purpose
core/stores/postgres.py	PostgresClient	connect(), close(), transaction(), insert_nodes_batch(), insert_edges_batch(), insert_document(), insert_outbox_batch(), fetch_pending_outbox(), update_outbox_status(), get_nodes_by_ids(), get_edges_by_ids(), init_schema(), health_check()	Async PostgreSQL client with connection pooling, batch inserts, outbox table operations, tenant‑aware queries, JSONB serialization, UUID conversion.
core/stores/neo4j.py	Neo4jClient	connect(), close(), create_constraints_and_indexes(), upsert_nodes_and_edges(), get_neighbors(), get_subgraph()	Async Neo4j client with chunked batch upserts (nodes then edges), parameterized Cypher queries, tenant isolation.
core/stores/qdrant.py	QdrantClient	connect(), close(), create_collection(), upsert_node_vectors(), upsert_edge_vectors(), upsert_batch_nodes(), upsert_batch_edges(), search_content_vectors(), search_evidence_vectors()	Async Qdrant client with named vectors (dense + sparse), batch upserts, payload filtering, collection creation.
core/stores/manager.py	StorageManager	begin_transaction(), stage_node(), stage_edge(), commit_transaction(), get_nodes(), get_edges(), get_neighbors()	Orchestrates the full indexing pipeline: staging, embedding generation, atomic PG write, outbox enqueue, and worker trigger.
core/stores/worker.py	OutboxWorker	process_batch(), _process_graph_entries(), _process_vector_entries()	Background worker that polls outbox table, processes graph and vector upserts separately, handles retries and failures.
core/stores/worker_daemon.py	run_worker()	-	Long‑lived daemon loop that continuously runs the OutboxWorker.
core/embeddings/service.py	EmbeddingService (ABC)	embed_text(), embed_image_path()	Abstract embedding service; we implemented DummyEmbeddingService for testing.
core/config.py	EngineConfig, DatabaseConfig, EmbeddingStrategy	-	Configuration for database connections, embedding strategies, active strategies.
core/models/	Node, Edge, Document, ProcessorManifest, BoundingBox, ModalityCategory, EdgeCategory, Granularity, DocStatus	-	Core data models (unchanged).
tests/test_storage.py	test_storage()	-	Basic integration test with 7 scenarios (success, outbox processing, PG/Neo4j/Qdrant verification, retry, multi‑tenant).
tests/test_storage_comprehensive.py	test_comprehensive_storage()	-	Extended test covering multiple modalities, edge categories, subgraphs, and deterministic embeddings.
How to use the complex infra (if you ever revisit):
Indexing a document:

python
manager = StorageManager(pg, neo4j, qdrant, embedder, config)
await manager.begin_transaction(tenant_id)
for node in nodes: manager.stage_node(node)
for edge in edges: manager.stage_edge(edge)
counts = await manager.commit_transaction(tenant_id)
# OutboxWorker runs asynchronously to write to Neo4j/Qdrant
Querying (retrieval):

pg.get_nodes_by_ids() and pg.get_edges_by_ids() for hydration.

qdrant.search_content_vectors() for vector search.

neo4j.get_neighbors() for graph traversal.