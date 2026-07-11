import json
import logging
import psycopg2
from psycopg2 import pool
from psycopg2.extras import execute_batch
from typing import List

from core.models.node import Node
from core.models.edge import Edge
from core.models.document import Document
from core.config import DatabaseConfig

logger = logging.getLogger(__name__)

class PostgresStore:
    def __init__(self, config: DatabaseConfig):
        try:
            # Using a connection pool prevents "connection closed" errors in async/multi-threaded environments
            self.pool = pool.SimpleConnectionPool(1, 10, dsn=config.postgres_dsn)
            logger.info("Postgres connection pool created successfully.")
            self._init_schema()
        except Exception as e:
            logger.error(f"Failed to initialize Postgres pool: {e}")
            raise

    def _get_conn(self):
        return self.pool.getconn()

    def _put_conn(self, conn):
        self.pool.putconn(conn)

    def _init_schema(self):
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS documents (
                        id UUID PRIMARY KEY,
                        tenant_id TEXT,
                        source_path TEXT,
                        plugin TEXT,
                        total_pages INT,
                        node_count INT,
                        edge_count INT,
                        subgraph_count INT,
                        status TEXT,
                        processing_log JSONB,
                        created_at TIMESTAMPTZ,
                        updated_at TIMESTAMPTZ,
                        is_deleted BOOLEAN
                    );
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS nodes (
                        id UUID PRIMARY KEY,
                        doc_id UUID,
                        tenant_id TEXT,
                        parent_id UUID,
                        subgraph_id UUID,
                        subgraph_role TEXT,
                        modality TEXT,
                        modality_category TEXT,
                        granularity TEXT,
                        sequence_index INT,
                        content TEXT,
                        content_hash TEXT,
                        bbox JSONB,
                        image_path TEXT,
                        embedding_refs JSONB,
                        node_meta JSONB,
                        created_at TIMESTAMPTZ,
                        updated_at TIMESTAMPTZ
                    );
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS edges (
                        id UUID PRIMARY KEY,
                        source_id UUID,
                        target_id UUID,
                        tenant_id TEXT,
                        type TEXT,
                        type_category TEXT,
                        is_bidirectional BOOLEAN,
                        weight FLOAT,
                        confidence FLOAT,
                        evidence TEXT,
                        edge_meta JSONB,
                        embedding_refs JSONB,
                        created_at TIMESTAMPTZ,
                        updated_at TIMESTAMPTZ
                    );
                """)
                conn.commit()
            logger.info("Postgres schema verified/initialized.")
        finally:
            self._put_conn(conn)

    def reset(self):
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("DROP TABLE IF EXISTS edges CASCADE;")
                cur.execute("DROP TABLE IF EXISTS nodes CASCADE;")
                cur.execute("DROP TABLE IF EXISTS documents CASCADE;")
            conn.commit()
            logger.info("Postgres tables dropped.")
            self._init_schema()
        finally:
            self._put_conn(conn)

    def save_document(self, doc: Document):
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO documents (id, tenant_id, source_path, plugin, total_pages, node_count, edge_count, subgraph_count, status, processing_log, created_at, updated_at, is_deleted)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        node_count = EXCLUDED.node_count,
                        edge_count = EXCLUDED.edge_count,
                        status = EXCLUDED.status,
                        processing_log = EXCLUDED.processing_log,
                        updated_at = EXCLUDED.updated_at;
                """, (
                    str(doc.id), doc.tenant_id, doc.source_path, doc.plugin, doc.total_pages,
                    doc.node_count, doc.edge_count, doc.subgraph_count, doc.status.value,
                    json.dumps(doc.processing_log), doc.created_at, doc.updated_at, doc.is_deleted
                ))
            conn.commit()
        finally:
            self._put_conn(conn)

    def save_nodes(self, nodes: List[Node]):
        if not nodes: return
        data = [
            (
                str(n.id), str(n.doc_id), n.tenant_id, str(n.parent_id) if n.parent_id else None,
                str(n.subgraph_id) if n.subgraph_id else None, n.subgraph_role, n.modality,
                n.modality_category.value, n.granularity.value if hasattr(n, 'granularity') and n.granularity else None, 
                n.sequence_index, n.content, n.content_hash, 
                json.dumps(n.bbox.model_dump()) if n.bbox else None, n.image_path,
                json.dumps(n.embedding_refs), json.dumps(n.node_meta), n.created_at, n.updated_at
            ) for n in nodes
        ]
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                execute_batch(cur, """
                    INSERT INTO nodes (id, doc_id, tenant_id, parent_id, subgraph_id, subgraph_role, modality, modality_category, granularity, sequence_index, content, content_hash, bbox, image_path, embedding_refs, node_meta, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        content = EXCLUDED.content,
                        content_hash = EXCLUDED.content_hash,
                        embedding_refs = EXCLUDED.embedding_refs,
                        node_meta = EXCLUDED.node_meta,
                        updated_at = EXCLUDED.updated_at;
                """, data)
            conn.commit()
            logger.info(f"Saved {len(nodes)} nodes to Postgres.")
        finally:
            self._put_conn(conn)

    def save_edges(self, edges: List[Edge]):
        if not edges: return
        data = [
            (
                str(e.id), str(e.source_id), str(e.target_id), e.tenant_id, e.type,
                e.type_category.value, e.is_bidirectional, getattr(e, 'weight', 1.0), 
                getattr(e, 'confidence', 1.0), e.evidence,
                json.dumps(e.edge_meta), json.dumps(e.embedding_refs), e.created_at, e.updated_at
            ) for e in edges
        ]
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                execute_batch(cur, """
                    INSERT INTO edges (id, source_id, target_id, tenant_id, type, type_category, is_bidirectional, weight, confidence, evidence, edge_meta, embedding_refs, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        evidence = EXCLUDED.evidence,
                        weight = EXCLUDED.weight,
                        confidence = EXCLUDED.confidence,
                        edge_meta = EXCLUDED.edge_meta,
                        embedding_refs = EXCLUDED.embedding_refs,
                        updated_at = EXCLUDED.updated_at;
                """, data)
            conn.commit()
            logger.info(f"Saved {len(edges)} edges to Postgres.")
        finally:
            self._put_conn(conn)

    def close(self):
        if self.pool and not self.pool.closed:
            self.pool.closeall()
            logger.info("Postgres connection pool closed.")