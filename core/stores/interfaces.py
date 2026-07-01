# core/stores/interfaces.py
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel
from core.models.node import Node, ModalityCategory
from core.models.edge import Edge, EdgeCategory
from core.models.document import Document
from core.models.processor import ProcessorManifest

class SearchResult(BaseModel):
    node_id: str
    score: float
    payload: Dict[str, Any]

class VectorStoreInterface(ABC):
    @abstractmethod
    def upsert_batch(self, items: List[Tuple[str, List[float], Dict]]) -> None: ...
    @abstractmethod
    def search(self, vector: List[float], top_k: int,
               tenant_id: str = "default",
               embedding_model: str = "bge_m3_v1",
               modality_categories: Optional[List[ModalityCategory]] = None,
               filter: Optional[Dict] = None) -> List[SearchResult]: ...
    @abstractmethod
    def delete_by_doc_id(self, doc_id: str, tenant_id: str = "default") -> None: ...

class GraphStoreInterface(ABC):
    @abstractmethod
    def add_nodes_batch(self, nodes: List[Node], tenant_id: str = "default") -> None: ...
    @abstractmethod
    def add_edges_batch(self, edges: List[Edge], tenant_id: str = "default") -> None: ...
    @abstractmethod
    def get_subgraph(self, container_node_id: str, tenant_id: str = "default") -> Tuple[List[Node], List[Edge]]: ...
    @abstractmethod
    def get_neighbors(self, node_id: str,
                      edge_categories: Optional[List[EdgeCategory]] = None,
                      depth: int = 1,
                      tenant_id: str = "default") -> List[Node]: ...
    @abstractmethod
    def get_path(self, source_id: str, target_id: str,
                 max_depth: int = 5,
                 tenant_id: str = "default") -> List[Edge]: ...
    @abstractmethod
    def delete_by_doc_id(self, doc_id: str, tenant_id: str = "default") -> None: ...

class MetaStoreInterface(ABC):
    @abstractmethod
    def create_document(self, doc: Document) -> str: ...
    @abstractmethod
    def update_document_status(self, doc_id: str, status: str) -> None: ...
    @abstractmethod
    def save_processor_manifest(self, manifest: ProcessorManifest) -> None: ...
    @abstractmethod
    def save_nodes_batch(self, nodes: List[Node]) -> None: ...
    @abstractmethod
    def save_edges_batch(self, edges: List[Edge]) -> None: ...
    @abstractmethod
    def get_nodes_by_ids(self, ids: List[str], tenant_id: str = "default") -> List[Node]: ...
    @abstractmethod
    def get_document_by_hash(self, content_hash: str, tenant_id: str = "default") -> Optional[Document]: ...
    @abstractmethod
    def list_documents(self, tenant_id: str = "default", limit: int = 100, offset: int = 0) -> List[Document]: ...
    @abstractmethod
    def delete_by_doc_id(self, doc_id: str) -> None: ...