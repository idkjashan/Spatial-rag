import uuid
from typing import List, Tuple
from core.models.node import Node, ModalityCategory, Granularity, BoundingBox
from core.models.edge import Edge, EdgeCategory
from core.models.document import Document

def generate_dummy_document(tenant_id: str = "default") -> Document:
    return Document(
        id=str(uuid.uuid4()),
        tenant_id=tenant_id,
        source_path="/dummy/doc.pdf",
        plugin="dummy",
    )

def generate_dummy_nodes_edges(doc_id: str, tenant_id: str = "default", num_nodes: int = 5, num_edges: int = 3) -> Tuple[List[Node], List[Edge]]:
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
            bbox=BoundingBox(x=random.random()*100, y=random.random()*100, w=10, h=10, page=1)
        )
        nodes.append(node)

    edges = []
    for i in range(min(num_edges, len(nodes)-1)):
        edge = Edge(
            id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            source_id=nodes[i].id,
            target_id=nodes[i+1].id,
            type="contains",
            type_category=EdgeCategory.HIERARCHY,
            weight=1.0,
            creator_processor="dummy",
            creator_version="1.0",
            evidence=f"Edge from {nodes[i].id} to {nodes[i+1].id}"
        )
        edges.append(edge)

    return nodes, edges