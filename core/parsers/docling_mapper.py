import os
import re
import uuid
import importlib.metadata
from typing import Dict, Tuple, List, Optional, Any

from core.models.node import Node, ModalityCategory, Granularity, BoundingBox
from core.models.edge import Edge, EdgeCategory
from core.models.document import Document, DocStatus

class DoclingMapper:
    LABEL_MAP = {
        "title":          {"modality": "title", "category": ModalityCategory.DOCUMENT_STRUCTURE, "is_container": True},
        "section_header": {"modality": "heading", "category": ModalityCategory.DOCUMENT_STRUCTURE, "is_container": True},
        "page_header":    {"modality": "page_header", "category": ModalityCategory.DOCUMENT_STRUCTURE},
        "page_footer":    {"modality": "page_footer", "category": ModalityCategory.DOCUMENT_STRUCTURE},
        "text":           {"modality": "paragraph", "category": ModalityCategory.TEXTUAL_CONTENT},
        "list_item":      {"modality": "list_item", "category": ModalityCategory.TEXTUAL_CONTENT},
        "caption":        {"modality": "caption", "category": ModalityCategory.TEXTUAL_CONTENT},
        "formula":        {"modality": "formula", "category": ModalityCategory.FORMULA},
        "picture":        {"modality": "figure", "category": ModalityCategory.IMAGE},
        "table":          {"modality": "table", "category": ModalityCategory.TABLE_CONTAINER, "is_container": True},
    }

    EDGE_MAP = {
        "doc_to_page":         ("contains_page", EdgeCategory.HIERARCHY),
        "heading_to_heading":  ("contains_section", EdgeCategory.HIERARCHY),
        "element_to_element":  ("contains", EdgeCategory.HIERARCHY),
        "element_to_caption":  ("captioned_by", EdgeCategory.CAPTION),
        "read_order":          ("read_order_next", EdgeCategory.READ_ORDER),
    }

    def __init__(self, doc_id: str, tenant_id: str, image_cache_path: str):
        self.doc_id = doc_id
        self.tenant_id = tenant_id
        self.image_cache_path = image_cache_path
        self.docling_doc = None
        
        try: self.processor_version = importlib.metadata.version("docling")
        except: self.processor_version = "unknown"
        
        self.id_map: Dict[str, str] = {}
        self.image_ref_map: Dict[str, Node] = {} 
        self.node_text_map: Dict[str, str] = {} 
        
        self.nodes: List[Node] = []
        self.edges: List[Edge] = []
        self.seq_idx = 0
        self.heading_stack = []
        # Map: page_no -> (width, height, dpi)
        self.page_dims_map: Dict[int, Tuple[float, float, float]] = {}

    def _safe_get_attr(self, obj: Any, attr_path: str, default: Any = None) -> Any:
        attrs = attr_path.split('.')
        for attr in attrs:
            if obj is None: return default
            obj = getattr(obj, attr, None)
        return obj if obj is not None else default

    def _get_ref_string(self, ref) -> str:
        if isinstance(ref, str): return ref
        for attr in ['cref', 'self_ref']:
            if hasattr(ref, attr): return getattr(ref, attr)
        return str(ref)

    def get_or_create_id(self, ref) -> str:
        ref_str = self._get_ref_string(ref)
        if ref_str not in self.id_map:
            self.id_map[ref_str] = str(uuid.uuid4())
        return self.id_map[ref_str]

    def _extract_bbox(self, item) -> Optional[BoundingBox]:
        """Robustly extracts, normalizes, and sanitizes bounding boxes."""
        bbox_data = None
        page_no = 1
        confidence = 1.0
        
        # 1. Try provenance (most common for text)
        if hasattr(item, 'prov') and item.prov:
            prov = item.prov[0]
            bbox_data = getattr(prov, 'bbox', None)
            page_no = getattr(prov, 'page_no', 1)
            confidence = getattr(prov, 'confidence', 1.0) # Confidence often lives on prov
            
        # 2. Fallback to direct bbox attribute (common for pictures)
        if not bbox_data and hasattr(item, 'bbox'):
            bbox_data = item.bbox
            page_no = getattr(item, 'page_no', 1)
            confidence = getattr(item, 'confidence', confidence)
            
        # 3. Fallback for tables (grid bbox)
        if not bbox_data and hasattr(item, 'data') and hasattr(item.data, 'grid') and hasattr(item.data.grid, 'bbox'):
            bbox_data = item.data.grid.bbox
            page_no = getattr(item, 'page_no', 1)
            confidence = getattr(bbox_data, 'confidence', confidence)

        if bbox_data:
            l = float(getattr(bbox_data, 'l', 0.0))
            t = float(getattr(bbox_data, 't', 0.0))
            r = float(getattr(bbox_data, 'r', 0.0))
            b = float(getattr(bbox_data, 'b', 0.0))
            
            # FIX: Ensure proper left/right and top/bottom ordering
            # Sometimes Docling returns coords where l > r or t > b depending on origin.
            l, r = min(l, r), max(l, r)
            t, b = min(t, b), max(t, b)
            
            page_w, page_h, dpi = self.page_dims_map.get(page_no, (0.0, 0.0, 72.0))
            
            if page_w > 0 and page_h > 0:
                # Normalize to 0.0-1.0 percentages
                x = l / page_w
                y = t / page_h
                w = (r - l) / page_w
                h = (b - t) / page_h
                
                # FIX: Clamp values strictly to [0.0, 1.0] to prevent spatial index errors
                x = max(0.0, min(1.0, x))
                y = max(0.0, min(1.0, y))
                w = max(0.0, min(1.0 - x, w))
                h = max(0.0, min(1.0 - y, h))
            else:
                # Fallback to absolute if dimensions missing (should ideally not happen)
                x, y = l, t
                w, h = r - l, b - t
            
            # FIX: Enforce minimum size to prevent zero-area geometries 
            # (breaks R-Tree/KD-Tree spatial queries later)
            min_dim = 0.001
            if w < min_dim: w = min_dim
            if h < min_dim: h = min_dim
                
            return BoundingBox(
                x=x, y=y, w=w, h=h, 
                page=int(page_no) - 1, 
                confidence=float(confidence), 
                dpi=float(dpi)
            )
        return None

    def _extract_ref_number(self, text: str) -> Optional[str]:
        match = re.search(r'(?:Figure|Fig\.|Table|Tab\.)\s*(\d+)', text, re.IGNORECASE)
        return match.group(1) if match else None

    def _get_hierarchy_parent(self, page_no: int, page_nodes_map: Dict) -> str:
        if self.heading_stack: return self.heading_stack[-1][1]
        page_node = page_nodes_map.get(page_no)
        return page_node.id if page_node else self.doc_id

    def _truncate_markdown_table(self, content: str) -> str:
        if len(content) <= Node.MAX_CONTENT_LENGTH: return content
        lines = content.split('\n')
        if len(lines) < 4: return content[:Node.MAX_CONTENT_LENGTH]
            
        header_lines = lines[:3]
        header_str = '\n'.join(header_lines)
        remaining_space = Node.MAX_CONTENT_LENGTH - len(header_str) - 50
        
        if remaining_space <= 0: return header_str[:Node.MAX_CONTENT_LENGTH]
            
        truncated_body = []
        current_len = 0
        for line in lines[3:]:
            if current_len + len(line) + 1 > remaining_space: break
            truncated_body.append(line)
            current_len += len(line) + 1
            
        return f"{header_str}\n{chr(10).join(truncated_body)}\n\n[... Table truncated for length ...]"

    def _create_node(self, ref, modality: str, mod_cat: ModalityCategory, content: str, 
                     bbox: Optional[BoundingBox], parent_id: str, subgraph_id: Optional[str], 
                     subgraph_role: str = "member", image_path: Optional[str] = None, 
                     fixed_id: Optional[str] = None, node_meta: Optional[Dict] = None) -> Node:
        self.seq_idx += 1
        ref_str = self._get_ref_string(ref)
        
        if fixed_id:
            node_id = fixed_id
            self.id_map[ref_str] = node_id
        else:
            node_id = self.get_or_create_id(ref_str)
            
        if mod_cat == ModalityCategory.TABLE_CONTAINER:
            content = self._truncate_markdown_table(content)
            
        node = Node(
            id=node_id, tenant_id=self.tenant_id, doc_id=self.doc_id, parent_id=parent_id,
            modality=modality, modality_category=mod_cat, content=content, bbox=bbox,
            granularity=Granularity.ELEMENT, sequence_index=self.seq_idx,
            subgraph_id=subgraph_id, subgraph_role=subgraph_role, image_path=image_path,
            node_meta=node_meta or {}, processor_name="docling", processor_version=self.processor_version
        )
        self.nodes.append(node)
        self.node_text_map[node_id] = content
        return node

    def _create_edge(self, source_id: str, target_id: str, edge_key: str, edge_meta: Optional[Dict] = None):
        edge_type, edge_cat = self.EDGE_MAP.get(edge_key, ("contains", EdgeCategory.HIERARCHY))
        self.edges.append(Edge(
            source_id=source_id, target_id=target_id, tenant_id=self.tenant_id,
            type=edge_type, type_category=edge_cat, creator_processor="docling",
            creator_version=self.processor_version, edge_meta=edge_meta or {}
        ))

    def map_document(self, docling_doc, source_path: str) -> Document:
        total_pages = len(docling_doc.pages) if hasattr(docling_doc, 'pages') else 0
        return Document(
            id=self.doc_id, tenant_id=self.tenant_id, source_path=source_path, plugin="docling",
            total_pages=total_pages, status=DocStatus.INDEXING, processor_name="docling", processor_version=self.processor_version
        )

    def map_nodes_and_edges(self, docling_doc) -> Tuple[List[Node], List[Edge]]:
        self.docling_doc = docling_doc
        page_nodes_map = self._process_pages(docling_doc)
        
        for item, level in docling_doc.iterate_items():
            if hasattr(item, 'label'):
                if item.label in ["title", "section_header", "page_header", "page_footer", "text", "list_item", "caption", "formula"]:
                    self._process_text_item(item, page_nodes_map)
                elif item.label == "table":
                    self._process_table_item(item, page_nodes_map)
                elif item.label == "picture":
                    self._process_picture_item(item, page_nodes_map)

        self._generate_read_order_edges()
        return self.nodes, self.edges

    def _process_pages(self, docling_doc) -> Dict[int, Node]:
        page_nodes_map = {}
        if hasattr(docling_doc, 'pages'):
            for page_no, page in docling_doc.pages.items():
                w = float(self._safe_get_attr(page, 'size.width', 0.0))
                h = float(self._safe_get_attr(page, 'size.height', 0.0))
                
                # Fallback to page bbox if size is missing
                if (w == 0.0 or h == 0.0) and hasattr(page, 'bbox') and page.bbox:
                    w = float(getattr(page.bbox, 'r', 0.0)) - float(getattr(page.bbox, 'l', 0.0))
                    h = float(getattr(page.bbox, 'b', 0.0)) - float(getattr(page.bbox, 't', 0.0))
                
                # Attempt to get DPI, default to 72.0
                dpi = float(self._safe_get_attr(page, 'dpi', 72.0))
                
                # Store dims for normalization later
                self.page_dims_map[page_no] = (w, h, dpi)
                
                page_id = str(uuid.uuid4())
                page_node = self._create_node(
                    ref=page_id, fixed_id=page_id, modality="page", mod_cat=ModalityCategory.DOCUMENT_STRUCTURE,
                    content=f"Page {page_no}", 
                    bbox=BoundingBox(x=0.0, y=0.0, w=1.0, h=1.0, page=page_no-1, confidence=1.0, dpi=dpi),
                    parent_id=self.doc_id, subgraph_id=None, subgraph_role="container", 
                    node_meta={"width": w, "height": h, "dpi": dpi}
                )
                page_nodes_map[page_no] = page_node
                self._create_edge(self.doc_id, page_node.id, "doc_to_page", edge_meta={"width": w, "height": h, "dpi": dpi})
        return page_nodes_map

    def _process_text_item(self, item, page_nodes_map):
        label = item.label
        config = self.LABEL_MAP.get(label, {"modality": "text", "category": ModalityCategory.TEXTUAL_CONTENT})
        
        bbox = self._extract_bbox(item)
        page_no = bbox.page + 1 if bbox else 1
        node_id = self.get_or_create_id(item.self_ref)
        node_meta, edge_meta = {}, {}

        # FIX: Safe cast to string
        text_content = str(item.text) if item.text else ""

        is_container = config.get("is_container", False)
        
        if is_container:
            level = 1 if label == "title" else getattr(item, 'level', 2)
            edge_meta['level'] = level
            while self.heading_stack and self.heading_stack[-1][0] >= level:
                self.heading_stack.pop()
            
            parent_id = self._get_hierarchy_parent(page_no, page_nodes_map)
            self.heading_stack.append((level, node_id, None))
            self._create_node(item.self_ref, config["modality"], config["category"], text_content, bbox, parent_id, None, "container", node_meta=node_meta)
            self._create_edge(parent_id, node_id, "heading_to_heading", edge_meta)
            
        elif label == "formula":
            parent_id = self._get_hierarchy_parent(page_no, page_nodes_map)
            edge_meta['display_type'] = getattr(item, 'display_type', 'inline')
            node_meta['latex'] = text_content
            self._create_node(item.self_ref, config["modality"], config["category"], text_content, bbox, parent_id, None, "member", node_meta=node_meta)
            self._create_edge(parent_id, node_id, "element_to_element", edge_meta)
            
        elif label == "list_item":
            parent_id = self._get_hierarchy_parent(page_no, page_nodes_map)
            marker = getattr(item, 'marker', None)
            if marker: node_meta['marker'] = marker
            self._create_node(item.self_ref, config["modality"], config["category"], text_content, bbox, parent_id, None, "member", node_meta=node_meta)
            self._create_edge(parent_id, node_id, "element_to_element", edge_meta)
            
        else:
            parent_id = self._get_hierarchy_parent(page_no, page_nodes_map)
            self._create_node(item.self_ref, config["modality"], config["category"], text_content, bbox, parent_id, None, "member", node_meta=node_meta)
            self._create_edge(parent_id, node_id, "element_to_element", edge_meta)

    def _process_table_item(self, item, page_nodes_map):
        bbox = self._extract_bbox(item)
        page_no = bbox.page + 1 if bbox else 1
        node_id = self.get_or_create_id(item.self_ref)
        parent_id = self._get_hierarchy_parent(page_no, page_nodes_map)
        
        content = item.export_to_markdown() if hasattr(item, 'export_to_markdown') else str(item)
        self._create_node(item.self_ref, "table", ModalityCategory.TABLE_CONTAINER, content, bbox, parent_id, node_id, "container")
        self._create_edge(parent_id, node_id, "element_to_element")
        self._link_captions(item, node_id, "table")

    def _process_picture_item(self, item, page_nodes_map):
        bbox = self._extract_bbox(item)
        page_no = bbox.page + 1 if bbox else 1
        node_id = self.get_or_create_id(item.self_ref)
        parent_id = self._get_hierarchy_parent(page_no, page_nodes_map)
        
        image_path = os.path.join(self.image_cache_path, f"{node_id}.png")
        node = self._create_node(item.self_ref, "figure", ModalityCategory.IMAGE, "", bbox, parent_id, None, "member", image_path)
        self.image_ref_map[item.self_ref] = node
        self._create_edge(parent_id, node_id, "element_to_element")
        
        try:
            if hasattr(item, 'get_image') and self.docling_doc:
                pil_image = item.get_image(doc=self.docling_doc)
                if pil_image: pil_image.save(image_path)
        except: pass
        
        self._link_captions(item, node_id, "figure")

    def _link_captions(self, item, source_node_id: str, ref_type: str):
        captions = getattr(item, 'captions', [])
        for cap_ref in captions:
            cap_ref_str = self._get_ref_string(cap_ref)
            if cap_ref_str in self.id_map:
                cap_node_id = self.id_map[cap_ref_str]
                cap_text = self.node_text_map.get(cap_node_id, "")
                ref_num = self._extract_ref_number(cap_text)
                
                meta = {"ref_type": ref_type}
                if ref_num: meta["number"] = ref_num
                self._create_edge(source_node_id, cap_node_id, "element_to_caption", meta)
                
                source_node = next((n for n in self.nodes if n.id == source_node_id), None)
                if source_node and cap_text:
                    source_node.content += f"\n\n[CAPTION]: {cap_text}" if source_node.content else f"[CAPTION]: {cap_text}"
                    self.node_text_map[source_node_id] = source_node.content

    def _generate_read_order_edges(self):
        allowed_categories = [
            ModalityCategory.TEXTUAL_CONTENT, ModalityCategory.FORMULA,
            ModalityCategory.TABLE_CONTAINER, ModalityCategory.IMAGE
        ]
        narrative_nodes = [n for n in self.nodes if n.modality_category in allowed_categories]
        sorted_nodes = sorted(narrative_nodes, key=lambda n: n.sequence_index)
        
        for i in range(len(sorted_nodes) - 1):
            self._create_edge(sorted_nodes[i].id, sorted_nodes[i+1].id, "read_order")