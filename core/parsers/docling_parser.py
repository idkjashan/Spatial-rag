import os
import uuid
from typing import Tuple, List
from pathlib import Path

from core.parsers.base import BaseParser
from core.parsers.docling_mapper import DoclingMapper
from core.models.document import Document, DocStatus
from core.models.node import Node, ModalityCategory
from core.models.edge import Edge

# Docling imports
from docling.document_converter import DocumentConverter
from docling.document_converter import PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

class DoclingParser(BaseParser):
    def __init__(self, tenant_id: str = "default", image_cache_path: str = "/tmp/spatialrag_images"):
        self.tenant_id = tenant_id
        self.image_cache_path = image_cache_path
        os.makedirs(self.image_cache_path, exist_ok=True)
        
        pipeline_options = PdfPipelineOptions(
            do_ocr=False, 
            do_table_structure=True,
            generate_page_images=False,
            generate_picture_images=True
        )
        
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

    def _extract_image_from_picture(self, pic_item, docling_doc, save_path: str) -> bool:
        """Robustly extracts image from a Docling PictureItem."""
        try:
            # In Docling v2, picture.image is an ImageRef. We need to get the actual PIL Image.
            if hasattr(pic_item, 'get_image'):
                # get_image returns a PIL Image.Image object
                image = pic_item.get_image(doc=docling_doc)
                if image:
                    image.save(save_path)
                    return True
                    
            # Fallback for older/special versions
            elif hasattr(pic_item, 'image') and pic_item.image:
                img_ref = pic_item.image
                # If it's already a PIL Image
                if hasattr(img_ref, 'save'):
                    img_ref.save(save_path)
                    return True
                # If it's an ImageRef with a URI
                elif hasattr(img_ref, 'uri') and img_ref.uri:
                    from PIL import Image
                    img = Image.open(str(img_ref.uri))
                    img.save(save_path)
                    return True
        except Exception as e:
            print(f"Image extraction warning: {e}")
        return False

    def parse(self, file_path: str) -> Tuple[Document, List[Node], List[Edge]]:
        doc_id = str(uuid.uuid4())
        
        result = self.converter.convert(file_path)
        docling_doc = result.document
        
        mapper = DoclingMapper(
            doc_id=doc_id,
            tenant_id=self.tenant_id,
            image_cache_path=self.image_cache_path
        )
        
        document = mapper.map_document(docling_doc, source_path=file_path)
        nodes, edges = mapper.map_nodes_and_edges(docling_doc)
        
        # Robust Image Extraction using the reference map
        if hasattr(docling_doc, 'pictures'):
            for pic_item in docling_doc.pictures:
                pic_ref = pic_item.self_ref
                if pic_ref in mapper.image_ref_map:
                    node = mapper.image_ref_map[pic_ref]
                    try:
                        success = self._extract_image_from_picture(pic_item, docling_doc, node.image_path)
                        if not success:
                            document.log(f"Failed to extract image for node {node.id}")
                    except Exception as e:
                        document.log(f"Failed to extract image for node {node.id}: {str(e)}")
        
        # Update document counts
        document.node_count = len(nodes)
        document.edge_count = len(edges)
        # Calculate number of subgraphs (containers)
        document.subgraph_count = sum(1 for n in nodes if n.subgraph_role == "container")
        
        document.status = DocStatus.READY
        document.log("Docling parsing completed successfully.")
        
        return document, nodes, edges