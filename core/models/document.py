# core/models/document.py
from core.models.base import SpatialRAGBase
from typing import List
from pydantic import Field
from enum import Enum

class DocStatus(str, Enum):
    PENDING = "pending"
    INDEXING = "indexing"
    EMBEDDING = "embedding"
    READY = "ready"
    PARTIAL = "partial"
    FAILED = "failed"

class Document(SpatialRAGBase):
    source_path: str
    plugin: str
    total_pages: int = 0
    total_duration_sec: float = 0.0

    node_count: int = 0
    edge_count: int = 0
    subgraph_count: int = 0

    status: DocStatus = DocStatus.PENDING
    processing_log: List[str] = Field(default_factory=list)

    def log(self, message: str) -> None:
        from datetime import datetime, timezone
        self.processing_log.append(f"[{datetime.now(timezone.utc).isoformat()}] {message}")
        self.touch()