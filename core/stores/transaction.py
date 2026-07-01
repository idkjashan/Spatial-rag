# core/stores/transaction.py
import logging
from contextlib import AbstractContextManager
from typing import List, Tuple, Dict, Any
from core.stores.interfaces import VectorStoreInterface, GraphStoreInterface, MetaStoreInterface
from core.models.node import Node, ModalityCategory
from core.models.edge import Edge
from core.models.document import Document, DocStatus
from core.models.processor import ProcessorManifest

logger = logging.getLogger(__name__)

class StorageTransaction(AbstractContextManager):
    def __init__(self, meta: MetaStoreInterface, graph: GraphStoreInterface,
                 vector: VectorStoreInterface, tenant_id: str = "default"):
        self.meta = meta
        self.graph = graph
        self.vector = vector
        self.tenant_id = tenant_id
        self.doc_id = None
        self.manifest = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.rollback()
        return False

    def begin(self, doc: Document, manifest: ProcessorManifest) -> str:
        doc.tenant_id = self.tenant_id
        manifest.tenant_id = self.tenant_id
        self.manifest = manifest
        self.meta.save_processor_manifest(manifest)
        self.doc_id = self.meta.create_document(doc)
        self.meta.update_document_status(self.doc_id, DocStatus.INDEXING.value)
        return self.doc_id

    def commit_nodes_and_edges(self, nodes: List[Node], edges: List[Edge]) -> None:
        structural_categories = {
            ModalityCategory.DOCUMENT_STRUCTURE,
            ModalityCategory.TEXTUAL_CONTENT,
            ModalityCategory.METADATA,
            ModalityCategory.ENTITY,
            ModalityCategory.VIDEO_FRAME,
            ModalityCategory.AUDIO_SEGMENT,
            ModalityCategory.TIMELINE_MARKER,
            ModalityCategory.RELATIONSHIP,
            ModalityCategory.IMAGE,
            ModalityCategory.SHAPE,
            ModalityCategory.FORMULA,
            ModalityCategory.EQUATION,
        }
        container_categories = {
            ModalityCategory.DIAGRAM_CONTAINER,
            ModalityCategory.TABLE_CONTAINER,
            ModalityCategory.CHART_CONTAINER,
            ModalityCategory.FLOWCHART_CONTAINER,
        }

        for node in nodes:
            node.tenant_id = self.tenant_id
            if node.modality_category in container_categories:
                if node.subgraph_id is not None:
                    raise ValueError(f"Container node {node.id} has subgraph_id. Must be None.")
                node.subgraph_role = "container"
            elif node.modality_category in structural_categories:
                node.subgraph_id = None
                node.subgraph_role = "member"
            else:
                if node.subgraph_id is None:
                    raise ValueError(f"Member node {node.id} has no subgraph_id.")
                node.subgraph_role = "member"
  
        for edge in edges:
            edge.tenant_id = self.tenant_id

        self.meta.save_nodes_batch(nodes)
        self.meta.save_edges_batch(edges)
        self.graph.add_nodes_batch(nodes, tenant_id=self.tenant_id)
        self.graph.add_edges_batch(edges, tenant_id=self.tenant_id)

    def commit_embeddings(self, vectors: List[Tuple[str, List[float], Dict]]) -> None:
        self.meta.update_document_status(self.doc_id, DocStatus.EMBEDDING.value)
        self.vector.upsert_batch(vectors)

    def finalize(self, status: str = DocStatus.READY.value) -> None:
        self.meta.update_document_status(self.doc_id, status)

    def rollback(self) -> None:
        if not self.doc_id:
            return
        errors = []
        try:
            self.meta.delete_by_doc_id(self.doc_id)
        except Exception as e:
            logger.error(f"Meta rollback failed: {e}")
            errors.append(f"Meta: {e}")
        try:
            self.graph.delete_by_doc_id(self.doc_id, tenant_id=self.tenant_id)
        except Exception as e:
            logger.error(f"Graph rollback failed: {e}")
            errors.append(f"Graph: {e}")
        try:
            self.vector.delete_by_doc_id(self.doc_id, tenant_id=self.tenant_id)
        except Exception as e:
            logger.error(f"Vector rollback failed: {e}")
            errors.append(f"Vector: {e}")
        if errors:
            raise RuntimeError(f"Partial rollback: {', '.join(errors)}")