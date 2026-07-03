# core/stores/postgres.py
import json
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncGenerator, Dict, List, Optional, Any

import asyncpg
from asyncpg import Pool, Connection

from core.models.node import Node
from core.models.edge import Edge
from core.models.document import Document

logger = logging.getLogger(__name__)


def _convert_uuid_to_str(data: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively convert UUID objects to strings."""
    for key, value in list(data.items()):
        if isinstance(value, uuid.UUID):
            data[key] = str(value)
        elif isinstance(value, dict):
            _convert_uuid_to_str(value)
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, uuid.UUID):
                    value[i] = str(item)
                elif isinstance(item, dict):
                    _convert_uuid_to_str(item)
    return data


class PostgresClient:
    def __init__(
        self,
        dsn: str,
        min_pool: int = 5,
        max_pool: int = 20,
        command_timeout: int = 30,
    ):
        self.dsn = dsn
        self.min_pool = min_pool
        self.max_pool = max_pool
        self.command_timeout = command_timeout
        self._pool: Optional[Pool] = None

    async def connect(self) -> None:
        if self._pool is not None:
            return
        self._pool = await asyncpg.create_pool(
            self.dsn,
            min_size=self.min_pool,
            max_size=self.max_pool,
            command_timeout=self.command_timeout,
        )
        logger.info(f"PostgreSQL pool established (min={self.min_pool}, max={self.max_pool})")

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("PostgreSQL pool closed")

    @asynccontextmanager
    async def transaction(self) -> AsyncGenerator[Connection, None]:
        if not self._pool:
            raise RuntimeError("Connection pool not initialized. Call connect() first.")
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                yield conn

    async def execute(self, query: str, *args) -> str:
        if not self._pool:
            raise RuntimeError("Connection pool not initialized.")
        async with self._pool.acquire() as conn:
            return await conn.execute(query, *args)

    async def fetch(self, query: str, *args) -> List[Dict[str, Any]]:
        if not self._pool:
            raise RuntimeError("Connection pool not initialized.")
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, *args)
            return [dict(row) for row in rows]

    async def fetch_one(self, query: str, *args) -> Optional[Dict[str, Any]]:
        if not self._pool:
            raise RuntimeError("Connection pool not initialized.")
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(query, *args)
            return dict(row) if row else None

    # ---- Document ----
    async def insert_document(self, doc: Document) -> None:
        query = """
            INSERT INTO documents (
                id, tenant_id, source_path, plugin, total_pages, total_duration_sec,
                node_count, edge_count, subgraph_count, status, processing_log,
                created_at, updated_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
            ON CONFLICT (id) DO UPDATE SET
                source_path = EXCLUDED.source_path,
                plugin = EXCLUDED.plugin,
                total_pages = EXCLUDED.total_pages,
                total_duration_sec = EXCLUDED.total_duration_sec,
                node_count = EXCLUDED.node_count,
                edge_count = EXCLUDED.edge_count,
                subgraph_count = EXCLUDED.subgraph_count,
                status = EXCLUDED.status,
                processing_log = EXCLUDED.processing_log,
                updated_at = NOW()
            WHERE documents.tenant_id = EXCLUDED.tenant_id
        """
        await self.execute(
            query,
            doc.id,
            doc.tenant_id,
            doc.source_path,
            doc.plugin,
            doc.total_pages,
            doc.total_duration_sec,
            doc.node_count,
            doc.edge_count,
            doc.subgraph_count,
            doc.status,
            json.dumps(doc.processing_log) if doc.processing_log else "[]",
            doc.created_at,
            doc.updated_at,
        )

    async def get_document(self, doc_id: str) -> Optional[Document]:
        row = await self.fetch_one("SELECT * FROM documents WHERE id = $1", doc_id)
        if row:
            if row.get('processing_log') and isinstance(row['processing_log'], str):
                row['processing_log'] = json.loads(row['processing_log'])
            # Convert UUIDs
            _convert_uuid_to_str(row)
            return Document(**row)
        return None

    # ---- Nodes ----
    async def insert_nodes_batch(self, nodes: List[Node]) -> None:
        if not nodes:
            return
        query = """
            INSERT INTO nodes (
                id, tenant_id, doc_id, parent_id, subgraph_id, subgraph_role,
                modality, modality_category, granularity, sequence_index,
                content, content_hash, content_truncated, bbox, image_path,
                start_timestamp, end_timestamp, embedding_refs, processor_name,
                processor_version, created_at, updated_at
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                $11, $12, $13, $14, $15, $16, $17, $18, $19,
                $20, $21, $22
            )
            ON CONFLICT (id) DO UPDATE SET
                doc_id = EXCLUDED.doc_id,
                parent_id = EXCLUDED.parent_id,
                subgraph_id = EXCLUDED.subgraph_id,
                subgraph_role = EXCLUDED.subgraph_role,
                modality = EXCLUDED.modality,
                modality_category = EXCLUDED.modality_category,
                granularity = EXCLUDED.granularity,
                sequence_index = EXCLUDED.sequence_index,
                content = EXCLUDED.content,
                content_hash = EXCLUDED.content_hash,
                content_truncated = EXCLUDED.content_truncated,
                bbox = EXCLUDED.bbox,
                image_path = EXCLUDED.image_path,
                start_timestamp = EXCLUDED.start_timestamp,
                end_timestamp = EXCLUDED.end_timestamp,
                embedding_refs = EXCLUDED.embedding_refs,
                processor_name = EXCLUDED.processor_name,
                processor_version = EXCLUDED.processor_version,
                updated_at = NOW()
            WHERE nodes.tenant_id = EXCLUDED.tenant_id
        """
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.executemany(
                    query,
                    [
                        (
                            n.id,
                            n.tenant_id,
                            n.doc_id,
                            n.parent_id,
                            n.subgraph_id,
                            n.subgraph_role,
                            n.modality,
                            n.modality_category.value if hasattr(n.modality_category, "value") else str(n.modality_category),
                            n.granularity.value if hasattr(n.granularity, "value") else str(n.granularity),
                            n.sequence_index,
                            n.content,
                            n.content_hash,
                            n.content_truncated,
                            json.dumps(n.bbox.model_dump()) if n.bbox else None,
                            n.image_path,
                            n.start_timestamp,
                            n.end_timestamp,
                            json.dumps(n.embedding_refs) if n.embedding_refs else "{}",
                            n.processor_name,
                            n.processor_version,
                            n.created_at,
                            n.updated_at,
                        )
                        for n in nodes
                    ],
                )

    # ---- Edges ----
    async def insert_edges_batch(self, edges: List[Edge]) -> None:
        if not edges:
            return
        query = """
            INSERT INTO edges (
                id, tenant_id, source_id, target_id, type, type_category,
                is_bidirectional, weight, confidence, evidence, edge_meta,
                embedding_refs, creator_processor, creator_version,
                created_at, updated_at
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16
            )
            ON CONFLICT (id) DO UPDATE SET
                source_id = EXCLUDED.source_id,
                target_id = EXCLUDED.target_id,
                type = EXCLUDED.type,
                type_category = EXCLUDED.type_category,
                is_bidirectional = EXCLUDED.is_bidirectional,
                weight = EXCLUDED.weight,
                confidence = EXCLUDED.confidence,
                evidence = EXCLUDED.evidence,
                edge_meta = EXCLUDED.edge_meta,
                embedding_refs = EXCLUDED.embedding_refs,
                creator_processor = EXCLUDED.creator_processor,
                creator_version = EXCLUDED.creator_version,
                updated_at = NOW()
            WHERE edges.tenant_id = EXCLUDED.tenant_id
        """
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.executemany(
                    query,
                    [
                        (
                            e.id,
                            e.tenant_id,
                            e.source_id,
                            e.target_id,
                            e.type,
                            e.type_category.value if hasattr(e.type_category, "value") else str(e.type_category),
                            e.is_bidirectional,
                            e.weight,
                            e.confidence,
                            e.evidence,
                            json.dumps(e.edge_meta) if e.edge_meta else "{}",
                            json.dumps(e.embedding_refs) if e.embedding_refs else "{}",
                            e.creator_processor,
                            e.creator_version,
                            e.created_at,
                            e.updated_at,
                        )
                        for e in edges
                    ],
                )

    # ---- Outbox ----
    async def insert_outbox_batch(self, outbox_entries: List[Dict[str, Any]]) -> None:
        if not outbox_entries:
            return
        query = """
            INSERT INTO txn_outbox (
                tenant_id, entity_id, entity_type, operation, payload,
                embedding_strategy, vector_data, idempotency_key,
                retry_count, max_retries, status, created_at
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12
            )
            ON CONFLICT (idempotency_key) DO NOTHING
        """
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.executemany(
                    query,
                    [
                        (
                            entry["tenant_id"],
                            entry["entity_id"],
                            entry["entity_type"],
                            entry["operation"],
                            json.dumps(entry["payload"]),
                            entry.get("embedding_strategy"),
                            json.dumps(entry.get("vector_data")) if entry.get("vector_data") else None,
                            entry["idempotency_key"],
                            entry.get("retry_count", 0),
                            entry.get("max_retries", 5),
                            entry.get("status", "PENDING"),
                            entry.get("created_at", datetime.now(timezone.utc)),
                        )
                        for entry in outbox_entries
                    ],
                )

    async def fetch_pending_outbox(self, tenant_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        query = """
            SELECT *
            FROM txn_outbox
            WHERE tenant_id = $1 AND status = 'PENDING'
            ORDER BY id ASC
            LIMIT $2
            FOR UPDATE SKIP LOCKED
        """
        return await self.fetch(query, tenant_id, limit)

    async def update_outbox_status(
        self,
        outbox_ids: List[int],
        status: str,
        error_msg: Optional[str] = None,
        increment_retry: bool = False,
    ) -> None:
        if not outbox_ids:
            return
        retry_clause = "retry_count = retry_count + 1" if increment_retry else ""
        if retry_clause:
            query = f"""
                UPDATE txn_outbox
                SET status = $1,
                    last_attempt_at = NOW(),
                    error_message = $2,
                    {retry_clause}
                WHERE id = ANY($3)
            """
        else:
            query = f"""
                UPDATE txn_outbox
                SET status = $1,
                    last_attempt_at = NOW(),
                    error_message = $2
                WHERE id = ANY($3)
            """
        await self.execute(query, status, error_msg, outbox_ids)

    # ---- Hydration helpers ----
    async def get_nodes_by_ids(self, tenant_id: str, node_ids: List[str]) -> List[Node]:
        if not node_ids:
            return []
        query = "SELECT * FROM nodes WHERE tenant_id = $1 AND id = ANY($2)"
        rows = await self.fetch(query, tenant_id, node_ids)
        converted = []
        for row in rows:
            # Parse JSONB fields if they are strings
            if row.get('bbox') and isinstance(row['bbox'], str):
                row['bbox'] = json.loads(row['bbox'])
            if row.get('embedding_refs') and isinstance(row['embedding_refs'], str):
                row['embedding_refs'] = json.loads(row['embedding_refs'])
            # Convert all UUID objects to strings
            _convert_uuid_to_str(row)
            converted.append(row)
        return [Node(**row) for row in converted]

    async def get_edges_by_ids(self, tenant_id: str, edge_ids: List[str]) -> List[Edge]:
        if not edge_ids:
            return []
        query = "SELECT * FROM edges WHERE tenant_id = $1 AND id = ANY($2)"
        rows = await self.fetch(query, tenant_id, edge_ids)
        converted = []
        for row in rows:
            if row.get('edge_meta') and isinstance(row['edge_meta'], str):
                row['edge_meta'] = json.loads(row['edge_meta'])
            if row.get('embedding_refs') and isinstance(row['embedding_refs'], str):
                row['embedding_refs'] = json.loads(row['embedding_refs'])
            # Convert UUIDs to strings
            _convert_uuid_to_str(row)
            converted.append(row)
        return [Edge(**row) for row in converted]

    # ---- Schema init ----
    async def init_schema(self) -> None:
        queries = [
            """CREATE TABLE IF NOT EXISTS documents (
                id UUID PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                source_path TEXT NOT NULL,
                plugin TEXT,
                total_pages INT DEFAULT 0,
                total_duration_sec FLOAT DEFAULT 0,
                node_count INT DEFAULT 0,
                edge_count INT DEFAULT 0,
                subgraph_count INT DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'PENDING',
                processing_log JSONB DEFAULT '[]',
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ
            )""",
            """CREATE TABLE IF NOT EXISTS nodes (
                id UUID PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                doc_id UUID REFERENCES documents(id) ON DELETE CASCADE,
                parent_id UUID,
                subgraph_id UUID,
                subgraph_role TEXT DEFAULT 'member',
                modality TEXT DEFAULT 'unknown',
                modality_category TEXT DEFAULT 'unknown',
                granularity TEXT DEFAULT 'element',
                sequence_index INT DEFAULT 0,
                content TEXT,
                content_hash TEXT,
                content_truncated BOOLEAN DEFAULT FALSE,
                bbox JSONB,
                image_path TEXT,
                start_timestamp FLOAT,
                end_timestamp FLOAT,
                embedding_refs JSONB DEFAULT '{}',
                processor_name TEXT NOT NULL,
                processor_version TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ
            )""",
            """CREATE TABLE IF NOT EXISTS edges (
                id UUID PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                source_id UUID REFERENCES nodes(id) ON DELETE CASCADE,
                target_id UUID REFERENCES nodes(id) ON DELETE CASCADE,
                type TEXT DEFAULT 'unknown',
                type_category TEXT DEFAULT 'unknown',
                is_bidirectional BOOLEAN DEFAULT FALSE,
                weight FLOAT DEFAULT 1.0,
                confidence FLOAT DEFAULT 1.0,
                evidence TEXT,
                edge_meta JSONB DEFAULT '{}',
                embedding_refs JSONB DEFAULT '{}',
                creator_processor TEXT DEFAULT 'core',
                creator_version TEXT DEFAULT '1.0.0',
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ
            )""",
            """CREATE TABLE IF NOT EXISTS processor_manifests (
                id UUID PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                processor_name TEXT NOT NULL,
                version TEXT NOT NULL,
                capability TEXT NOT NULL,
                model_parameters JSONB DEFAULT '{}',
                is_active BOOLEAN DEFAULT TRUE,
                loaded_memory_mb FLOAT,
                inputs_required JSONB DEFAULT '[]',
                outputs_produced JSONB DEFAULT '[]',
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ,
                UNIQUE(tenant_id, processor_name, version)
            )""",
            """CREATE TABLE IF NOT EXISTS txn_outbox (
                id BIGSERIAL PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                entity_id UUID NOT NULL,
                entity_type TEXT NOT NULL,
                operation TEXT NOT NULL,
                payload JSONB NOT NULL,
                embedding_strategy TEXT,
                vector_data JSONB,
                idempotency_key TEXT UNIQUE NOT NULL,
                retry_count INT DEFAULT 0,
                max_retries INT DEFAULT 5,
                status TEXT DEFAULT 'PENDING',
                error_message TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                last_attempt_at TIMESTAMPTZ
            )""",
            """CREATE TABLE IF NOT EXISTS graph_statistics (
                tenant_id TEXT NOT NULL,
                stat_key TEXT NOT NULL,
                stat_value JSONB NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY (tenant_id, stat_key)
            )""",
            "CREATE INDEX IF NOT EXISTS idx_nodes_tenant_doc ON nodes (tenant_id, doc_id)",
            "CREATE INDEX IF NOT EXISTS idx_nodes_tenant_subgraph ON nodes (tenant_id, subgraph_id)",
            "CREATE INDEX IF NOT EXISTS idx_nodes_tenant_modality ON nodes (tenant_id, modality_category)",
            "CREATE INDEX IF NOT EXISTS idx_nodes_content_hash ON nodes (content_hash)",
            "CREATE INDEX IF NOT EXISTS idx_edges_tenant_source ON edges (tenant_id, source_id)",
            "CREATE INDEX IF NOT EXISTS idx_edges_tenant_target ON edges (tenant_id, target_id)",
            "CREATE INDEX IF NOT EXISTS idx_edges_tenant_category ON edges (tenant_id, type_category)",
            "CREATE INDEX IF NOT EXISTS idx_outbox_tenant_status ON txn_outbox (tenant_id, status) WHERE status = 'PENDING'",
            "CREATE INDEX IF NOT EXISTS idx_outbox_idempotency ON txn_outbox (idempotency_key)",
        ]
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                for q in queries:
                    try:
                        await conn.execute(q)
                    except asyncpg.exceptions.DuplicateTableError:
                        pass
                    except Exception as e:
                        logger.warning(f"Schema init warning for query: {q[:60]}... Error: {e}")
        logger.info("PostgreSQL schema initialized.")

    async def health_check(self) -> bool:
        try:
            await self.execute("SELECT 1")
            return True
        except Exception:
            return False