import os
import re
import uuid
import logging
import importlib.metadata
from typing import Dict, Tuple, List, Optional, Any

from core.models.node import Node, ModalityCategory, Granularity, BoundingBox
from core.models.edge import Edge, EdgeCategory
from core.models.document import Document, DocStatus

logger = logging.getLogger(__name__)

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
        "table_to_row":        ("contains_row", EdgeCategory.TABLE_HIERARCHY),
        "list_to_item":        ("contains_item", EdgeCategory.HIERARCHY),
        "eq_to_latex":         ("contains_latex", EdgeCategory.HIERARCHY),
    }

    def __init__(self, doc_id: str, tenant_id: str, image_cache_path: str):
        self.doc_id = doc_id
        self.tenant_id = tenant_id
        self.image_cache_path = image_cache_path
        self.docling_doc = None
        
        try: 
            self.processor_version = importlib.metadata.version("docling")
        except Exception: 
            self.processor_version = "unknown"
            
        self.id_map: Dict[str, str] = {}
        self.image_ref_map: Dict[str, Node] = {} 
        self.node_text_map: Dict[str, str] = {} 
        
        self.nodes: List[Node] = []
        self.edges: List[Edge] = []
        self.seq_idx = 0
        self.heading_stack = []
        self.page_dims_map: Dict[int, Tuple[float, float, float]] = {}
        
        self.deferred_captions: Dict[str, List[str]] = {}
        self.current_list_id: Optional[str] = None
        logger.info(f"DoclingMapper initialized for doc_id: {doc_id}")

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
        bbox_data = None
        page_no = 1
        confidence = 1.0
        
        if hasattr(item, 'prov') and item.prov:
            prov = item.prov[0]
            bbox_data = getattr(prov, 'bbox', None)
            page_no = getattr(prov, 'page_no', 1)
            confidence = getattr(prov, 'confidence', 1.0)
            
        if not bbox_data and hasattr(item, 'bbox'):
            bbox_data = item.bbox
            page_no = getattr(item, 'page_no', 1)
            confidence = getattr(item, 'confidence', confidence)
            
        if not bbox_data and hasattr(item, 'data') and hasattr(item.data, 'grid') and hasattr(item.data.grid, 'bbox'):
            bbox_data = item.data.grid.bbox
            page_no = getattr(item, 'page_no', 1)
            confidence = getattr(bbox_data, 'confidence', confidence)

        if bbox_data:
            l = float(getattr(bbox_data, 'l', 0.0))
            t = float(getattr(bbox_data, 't', 0.0))
            r = float(getattr(bbox_data, 'r', 0.0))
            b = float(getattr(bbox_data, 'b', 0.0))
            
            l, r = min(l, r), max(l, r)
            t, b = min(t, b), max(t, b)
            
            page_w, page_h, dpi = self.page_dims_map.get(page_no, (0.0, 0.0, 72.0))
            
            if page_w > 0 and page_h > 0:
                x = l / page_w
                y = t / page_h
                w = (r - l) / page_w
                h = (b - t) / page_h
                x = max(0.0, min(1.0, x))
                y = max(0.0, min(1.0, y))
                w = max(0.0, min(1.0 - x, w))
                h = max(0.0, min(1.0 - y, h))
            else:
                x, y = l, t
                w, h = r - l, b - t
            
            min_dim = 0.001
            if w < min_dim: w = min_dim
            if h < min_dim: h = min_dim
                
            return BoundingBox(x=x, y=y, w=w, h=h, page=int(page_no) - 1, confidence=float(confidence), dpi=float(dpi))
        return None

    def _extract_ref_number(self, text: str) -> Optional[str]:
        match = re.search(r'(?:Figure|Fig\.|Table|Tab\.|Equation|Eq\.|Section|Sec\.)\s*(\d+(?:\.\d+)*)', text, re.IGNORECASE)
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
                     fixed_id: Optional[str] = None, node_meta: Optional[Dict] = None,
                     granularity: Granularity = Granularity.ELEMENT) -> Node:
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
            granularity=granularity, sequence_index=self.seq_idx,
            subgraph_id=subgraph_id, subgraph_role=subgraph_role, image_path=image_path,
            node_meta=node_meta or {}, processor_name="docling", processor_version=self.processor_version
        )
        self.nodes.append(node)
        self.node_text_map[node_id] = content
        logger.debug(f"Created Node: {modality} ({granularity.value}) - ID: {node_id}")
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
        logger.info(f"Mapping document with {total_pages} pages.")
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

        self._inject_deferred_captions()
        self._generate_read_order_edges()
        logger.info(f"Mapping complete. Total Nodes: {len(self.nodes)}, Total Edges: {len(self.edges)}")
        return self.nodes, self.edges

    def _process_pages(self, docling_doc) -> Dict[int, Node]:
        page_nodes_map = {}
        if hasattr(docling_doc, 'pages'):
            for page_no, page in docling_doc.pages.items():
                w = float(self._safe_get_attr(page, 'size.width', 0.0))
                h = float(self._safe_get_attr(page, 'size.height', 0.0))
                
                if (w == 0.0 or h == 0.0) and hasattr(page, 'bbox') and page.bbox:
                    w = float(getattr(page.bbox, 'r', 0.0)) - float(getattr(page.bbox, 'l', 0.0))
                    h = float(getattr(page.bbox, 'b', 0.0)) - float(getattr(page.bbox, 't', 0.0))
                
                dpi = float(self._safe_get_attr(page, 'dpi', 72.0))
                self.page_dims_map[page_no] = (w, h, dpi)
                
                page_id = str(uuid.uuid4())
                page_node = self._create_node(
                    ref=page_id, fixed_id=page_id, modality="page", mod_cat=ModalityCategory.DOCUMENT_STRUCTURE,
                    content=f"Page {page_no}", 
                    bbox=BoundingBox(x=0.0, y=0.0, w=1.0, h=1.0, page=page_no-1, confidence=1.0, dpi=dpi),
                    parent_id=self.doc_id, subgraph_id=None, subgraph_role="container", 
                    node_meta={"width": w, "height": h, "dpi": dpi},
                    granularity=Granularity.PAGE
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

        text_content = str(item.text) if item.text else ""
        is_container = config.get("is_container", False)
        
        if is_container:
            self.current_list_id = None 
            level = 1 if label == "title" else getattr(item, 'level', 2)
            edge_meta['level'] = level
            while self.heading_stack and self.heading_stack[-1][0] >= level:
                self.heading_stack.pop()
            
            parent_id = self._get_hierarchy_parent(page_no, page_nodes_map)
            self.heading_stack.append((level, node_id, None))
            self._create_node(item.self_ref, config["modality"], config["category"], text_content, bbox, parent_id, None, "container", node_meta=node_meta, granularity=Granularity.SECTION)
            self._create_edge(parent_id, node_id, "heading_to_heading", edge_meta)
            
        elif label == "formula":
            self.current_list_id = None
            parent_id = self._get_hierarchy_parent(page_no, page_nodes_map)
            
            eq_id = self.get_or_create_id(item.self_ref)
            self._create_node(item.self_ref, "equation", ModalityCategory.EQUATION, "Mathematical Equation", bbox, parent_id, eq_id, "container", granularity=Granularity.BLOCK)
            self._create_edge(parent_id, eq_id, "element_to_element", edge_meta)
            
            formula_ref = str(uuid.uuid4())
            image_path = os.path.join(self.image_cache_path, f"{formula_ref}.png")
            
            try:
                pil_image = None
                if hasattr(item, 'get_image') and self.docling_doc:
                    try:
                        pil_image = item.get_image(doc=self.docling_doc)
                    except Exception as e:
                        logger.debug(f"Direct image extraction failed for formula, falling back to crop: {e}")
                        pil_image = None
                
                if not pil_image and bbox and self.docling_doc:
                    page_no_int = bbox.page + 1
                    page = self.docling_doc.pages.get(page_no_int)
                    if page and hasattr(page, 'get_image'):
                        page_img = page.get_image()
                        if page_img:
                            w, h = page_img.size
                            left = int(bbox.x * w)
                            top = int(bbox.y * h)
                            right = int((bbox.x + bbox.w) * w)
                            bottom = int((bbox.y + bbox.h) * h)
                            pil_image = page_img.crop((left, top, right, bottom))
                
                if pil_image:
                    pil_image.save(image_path)
                else:
                    image_path = None 
            except Exception as e:
                logger.error(f"Failed to extract formula image: {e}")
                image_path = None
            
            formula_node = self._create_node(
                ref=formula_ref, fixed_id=formula_ref, modality="formula", mod_cat=ModalityCategory.FORMULA, 
                content=text_content, bbox=bbox, parent_id=eq_id, subgraph_id=eq_id, subgraph_role="member", 
                image_path=image_path, node_meta={"latex": text_content},
                granularity=Granularity.ELEMENT
            )
            self._create_edge(eq_id, formula_node.id, "eq_to_latex")
            
        elif label == "list_item":
            if self.current_list_id is None:
                list_ref = str(uuid.uuid4())
                parent_id = self._get_hierarchy_parent(page_no, page_nodes_map)
                list_node = self._create_node(list_ref, "list", ModalityCategory.DOCUMENT_STRUCTURE, "List Group", bbox, parent_id, None, "container", granularity=Granularity.BLOCK)
                list_node.subgraph_id = list_node.id
                self.current_list_id = list_node.id
                self._create_edge(parent_id, list_node.id, "element_to_element")
            else:
                parent_id = self.current_list_id
                
            marker = getattr(item, 'marker', None)
            if marker: node_meta['marker'] = marker
            list_item_node = self._create_node(item.self_ref, config["modality"], config["category"], text_content, bbox, parent_id, self.current_list_id, "member", node_meta=node_meta, granularity=Granularity.ELEMENT)
            self._create_edge(parent_id, list_item_node.id, "list_to_item", edge_meta)
            
        else:
            self.current_list_id = None
            parent_id = self._get_hierarchy_parent(page_no, page_nodes_map)
            gran = Granularity.ELEMENT if label == "caption" else Granularity.BLOCK
            self._create_node(item.self_ref, config["modality"], config["category"], text_content, bbox, parent_id, None, "member", node_meta=node_meta, granularity=gran)
            self._create_edge(parent_id, node_id, "element_to_element", edge_meta)

    def _process_table_item(self, item, page_nodes_map):
        self.current_list_id = None
        bbox = self._extract_bbox(item)
        page_no = bbox.page + 1 if bbox else 1
        node_id = self.get_or_create_id(item.self_ref)
        parent_id = self._get_hierarchy_parent(page_no, page_nodes_map)
        
        content = item.export_to_markdown() if hasattr(item, 'export_to_markdown') else str(item)
        self._create_node(item.self_ref, "table", ModalityCategory.TABLE_CONTAINER, content, bbox, parent_id, node_id, "container", granularity=Granularity.BLOCK)
        self._create_edge(parent_id, node_id, "element_to_element")
        
        self._deconstruct_table(content, node_id)
        self._link_captions(item, node_id, "table")

    def _deconstruct_table(self, markdown_content: str, container_id: str):
        lines = [l.strip() for l in markdown_content.strip().split('\n') if l.strip()]
        if len(lines) < 2: return
            
        headers = [h.strip() for h in lines[0].split('|') if h.strip()]
        if not headers: return
        
        data_lines = lines[2:] if re.match(r'^\s*\|?[\s:-]+\|[\s:\-]+\|?\s*$', lines[1]) else lines[1:]
        
        row_idx = 0
        for line in data_lines:
            if not line.startswith('|'): continue
            vals = [v.strip() for v in line.split('|') if v.strip() != '']
            if not vals: continue
            
            row_meta = {"row_index": row_idx}
            for i, val in enumerate(vals):
                col_name = headers[i] if i < len(headers) else f"col_{i}"
                row_meta[col_name] = val
                
            row_content = " | ".join(vals)
            row_ref = str(uuid.uuid4())
            
            row_node = self._create_node(
                ref=row_ref, fixed_id=row_ref, 
                modality="table_row", mod_cat=ModalityCategory.TABLE_CONTENT, 
                content=row_content, bbox=None, 
                parent_id=container_id, subgraph_id=container_id, subgraph_role="member", 
                node_meta=row_meta,
                granularity=Granularity.ELEMENT
            )
            self._create_edge(container_id, row_node.id, "table_to_row", {"row_index": row_idx})
            row_idx += 1

    def _process_picture_item(self, item, page_nodes_map):
        self.current_list_id = None
        bbox = self._extract_bbox(item)
        page_no = bbox.page + 1 if bbox else 1
        node_id = self.get_or_create_id(item.self_ref)
        parent_id = self._get_hierarchy_parent(page_no, page_nodes_map)
        
        image_path = os.path.join(self.image_cache_path, f"{node_id}.png")
        node = self._create_node(item.self_ref, "figure", ModalityCategory.IMAGE, "Figure image.", bbox, parent_id, None, "member", image_path, granularity=Granularity.BLOCK)
        self.image_ref_map[item.self_ref] = node
        self._create_edge(parent_id, node_id, "element_to_element")
        
        try:
            if hasattr(item, 'get_image') and self.docling_doc:
                pil_image = item.get_image(doc=self.docling_doc)
                if pil_image: pil_image.save(image_path)
        except Exception as e:
            logger.debug(f"Could not extract/save image for node {node_id}: {e}")
        
        self._link_captions(item, node_id, "figure")

    def _link_captions(self, item, source_node_id: str, ref_type: str):
        captions = getattr(item, 'captions', [])
        for cap_ref in captions:
            cap_ref_str = self._get_ref_string(cap_ref)
            cap_node_id = self.get_or_create_id(cap_ref_str)
            self._create_edge(source_node_id, cap_node_id, "element_to_caption", {"ref_type": ref_type})
            
            if source_node_id not in self.deferred_captions:
                self.deferred_captions[source_node_id] = []
            self.deferred_captions[source_node_id].append(cap_node_id)

    def _inject_deferred_captions(self):
        for source_id, cap_ids in self.deferred_captions.items():
            source_node = next((n for n in self.nodes if n.id == source_id), None)
            if not source_node: continue
            
            for cap_id in cap_ids:
                cap_text = self.node_text_map.get(cap_id, "")
                if cap_text:
                    if "[CAPTION]:" not in source_node.content:
                        source_node.content += f"\n\n[CAPTION]: {cap_text}"
                        self.node_text_map[source_id] = source_node.content
                    
                    ref_num = self._extract_ref_number(cap_text)
                    if ref_num:
                        for e in self.edges:
                            if e.source_id == source_id and e.target_id == cap_id and e.type == "captioned_by":
                                e.edge_meta["number"] = ref_num
                                break

    def _generate_read_order_edges(self):
        narrative_modalities = ["paragraph", "table", "figure", "equation", "list"]
        narrative_nodes = [n for n in self.nodes if n.modality in narrative_modalities]
        sorted_nodes = sorted(narrative_nodes, key=lambda n: n.sequence_index)
        
        for i in range(len(sorted_nodes) - 1):
            self._create_edge(sorted_nodes[i].id, sorted_nodes[i+1].id, "read_order")