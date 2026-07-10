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
            generate_page_images=True,  # FIX: Required to crop formula images
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
            if hasattr(pic_item, 'get_image'):
                image = pic_item.get_image(doc=docling_doc)
                if image:
                    image.save(save_path)
                    return True
                    
            elif hasattr(pic_item, 'image') and pic_item.image:
                img_ref = pic_item.image
                if hasattr(img_ref, 'save'):
                    img_ref.save(save_path)
                    return True
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
        
        # FIX: Crop Formula Images from Page Images
        if hasattr(docling_doc, 'pages'):
            for node in nodes:
                # Look for child formula nodes that have a bbox but no image path yet
                if node.modality == "formula" and node.bbox and not node.image_path:
                    page_no = node.bbox.page + 1
                    page = docling_doc.pages.get(page_no)
                    if page and hasattr(page, 'get_image'):
                        try:
                            page_img = page.get_image()
                            if page_img:
                                w, h = page_img.size
                                left = int(node.bbox.x * w)
                                top = int(node.bbox.y * h)
                                right = int((node.bbox.x + node.bbox.w) * w)
                                bottom = int((node.bbox.y + node.bbox.h) * h)
                                
                                # Crop and save
                                formula_img = page_img.crop((left, top, right, bottom))
                                img_path = os.path.join(self.image_cache_path, f"{node.id}.png")
                                formula_img.save(img_path)
                                node.image_path = img_path
                        except Exception as e:
                            document.log(f"Failed to crop formula image for node {node.id}: {str(e)}")
        
        # Update document counts
        document.node_count = len(nodes)
        document.edge_count = len(edges)
        document.subgraph_count = sum(1 for n in nodes if n.subgraph_role == "container")
        
        document.status = DocStatus.READY
        document.log("Docling parsing completed successfully.")
        
        return document, nodes, edges