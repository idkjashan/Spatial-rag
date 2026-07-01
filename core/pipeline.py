# core/pipeline.py
from core.stores.transaction import StorageTransaction
from core.models.document import Document, DocStatus
from core.models.processor import ProcessorManifest, ProcessorCapability
from core.config import EngineConfig
from core.stores.interfaces import MetaStoreInterface, GraphStoreInterface, VectorStoreInterface

# ---- Placeholder functions ----
def docling_parse(file_path):
    """Placeholder: returns a list of elements with .type and .content."""
    class DummyElement:
        def __init__(self, t, c):
            self.type = t
            self.content = c
    return [DummyElement("text", "Sample text"), DummyElement("figure", "image data")]

def is_complex_diagram(entity):
    """Placeholder: returns True if the entity is a complex diagram."""
    return "diagram" in entity.type

# ---- Placeholder processor classes ----
class TextProcessor:
    @staticmethod
    def process(entity, config):
        # Returns (nodes, edges)
        return [], []

class TableProcessor:
    @staticmethod
    def process(entity, config):
        return [], []

class DiagramProcessor:
    @staticmethod
    def process(entity, config):
        return [], []

class VLMProcessor:
    @staticmethod
    def process(entity, config):
        return [], []

# ---- Main pipeline ----
def index_document(
    raw_file_path: str,
    config: EngineConfig,
    meta: MetaStoreInterface,
    graph: GraphStoreInterface,
    vector: VectorStoreInterface,
    embedder,  # injected embedder with .embed_text() and .embed_image(path)
) -> str:
    docling_output = docling_parse(raw_file_path)

    doc = Document(
        source_path=raw_file_path,
        plugin="docling_v1",
        tenant_id=config.tenant_id
    )
    manifest = ProcessorManifest(
        processor_name="docling_v2",
        version="1.0.0",
        capability=ProcessorCapability.TEXT_PARSER,
        inputs_required=["pdf_file"],
        outputs_produced=["parsed_elements"]
    )

    tx = StorageTransaction(meta, graph, vector, tenant_id=config.tenant_id)
    with tx:
        doc_id = tx.begin(doc, manifest)
        all_nodes = []
        all_edges = []

        for entity in docling_output:
            if entity.type == "text":
                nodes, edges = TextProcessor.process(entity, config)
            elif entity.type == "table":
                nodes, edges = TableProcessor.process(entity, config)
            elif entity.type == "figure" and is_complex_diagram(entity):
                nodes, edges = DiagramProcessor.process(entity, config)
            elif entity.type == "figure":
                nodes, edges = VLMProcessor.process(entity, config)
            else:
                continue
            all_nodes.extend(nodes)
            all_edges.extend(edges)

        tx.commit_nodes_and_edges(all_nodes, all_edges)

        vectors = []
        for node in all_nodes:
            # text embedding
            vec_text = embedder.embed_text(node.content)
            vectors.append((node.id, vec_text, {"model": "bge_m3", "type": "text"}))

            # visual embedding if image_path exists
            if node.image_path:
                vec_vis = embedder.embed_image(node.image_path)
                vis_id = f"{node.id}_clip"
                vectors.append((vis_id, vec_vis, {"model": "clip_vit", "type": "visual"}))
                node.embedding_refs["clip_vit_v1"] = vis_id

        tx.commit_embeddings(vectors)
        tx.finalize(DocStatus.READY.value)

        return doc_id