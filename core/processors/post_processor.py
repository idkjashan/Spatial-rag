import re
from typing import List, Dict, Tuple, Optional
from shapely.geometry import box

from core.models.node import Node, ModalityCategory
from core.models.edge import Edge, EdgeCategory
from core.models.document import Document

class GraphPostProcessor:
    """
    Phase 0.75 Post-Processor:
    Executes deterministic spatial, referential, and contextual enrichment.
    Defers LLM/semantic extraction to Phase 2.
    """
    
    SPATIAL_CATEGORIES = [
        ModalityCategory.TEXTUAL_CONTENT,
        ModalityCategory.TABLE_CONTAINER,
        ModalityCategory.IMAGE,
        ModalityCategory.FORMULA
    ]
    
    REFERENCE_REGEX = re.compile(
        r'\b(?:Figure|Fig\.?|Table|Tab\.?|Equation|Eq\.?|Section|Sec\.?)\s*(\d+(?:\.\d+)*)\b', 
        re.IGNORECASE
    )
    
    PAGE_REFERENCE_REGEX = re.compile(
        r'\b(?:Page|Pg\.?|P\.)\s*(\d+)\b', 
        re.IGNORECASE
    )

    def __init__(self, adjacency_threshold: float = 0.05, band_overlap: float = 0.5):
        self.adjacency_threshold = adjacency_threshold
        self.band_overlap = band_overlap

    def process(self, document: Document, nodes: List[Node], edges: List[Edge]) -> Tuple[Document, List[Node], List[Edge]]:
        node_map = {n.id: n for n in nodes}
        ref_lookup = self._build_ref_lookup(nodes)
        
        self._augment_context(nodes, node_map)
        self._resolve_cross_references(nodes, edges, ref_lookup)
        self._extract_spatial_relations(nodes, edges)
        self._generate_edge_evidence(edges, node_map)
        
        document.edge_count = len(edges)
        document.touch()
        return document, nodes, edges

    def _build_ref_lookup(self, nodes: List[Node]) -> Dict[str, Dict[str, str]]:
        ref_lookup: Dict[str, Dict[str, str]] = {
            "figure": {}, "table": {}, "formula": {}, "section": {}, "page": {}
        }
        for node in nodes:
            if node.modality_category == ModalityCategory.IMAGE:
                match = self.REFERENCE_REGEX.search(node.content)
                if match: ref_lookup["figure"][match.group(1)] = node.id
            elif node.modality_category == ModalityCategory.TABLE_CONTAINER:
                match = self.REFERENCE_REGEX.search(node.content)
                if match: ref_lookup["table"][match.group(1)] = node.id
            elif node.modality_category == ModalityCategory.FORMULA:
                match = self.REFERENCE_REGEX.search(node.content)
                if match: ref_lookup["formula"][match.group(1)] = node.id
            elif node.modality_category == ModalityCategory.DOCUMENT_STRUCTURE and node.modality in ['title', 'heading']:
                match = re.match(r'^(\d+(?:\.\d+)*)\s+', node.content)
                if match:
                    ref_lookup["section"][match.group(1)] = node.id
                else:
                    match = self.REFERENCE_REGEX.search(node.content)
                    if match: ref_lookup["section"][match.group(1)] = node.id
            elif node.modality == "page" and node.bbox:
                ref_lookup["page"][str(node.bbox.page + 1)] = node.id
        return ref_lookup

    def _augment_context(self, nodes: List[Node], node_map: Dict[str, Node]):
        for node in nodes:
            if node.modality_category in [ModalityCategory.TEXTUAL_CONTENT, 
                                          ModalityCategory.TABLE_CONTAINER, 
                                          ModalityCategory.FORMULA,
                                          ModalityCategory.IMAGE]:
                
                if node.content.startswith("[SECTION_CONTEXT]"):
                    continue
                    
                context_path = self._get_context_path(node, node_map)
                # FIX: Only prepend context if the node actually has content!
                # This prevents tricking the Enricher into thinking the context path is the LaTeX text.
                if context_path and node.content.strip():
                    node.content = f"[SECTION_CONTEXT] {context_path} > {node.content}"

    def _get_context_path(self, node: Node, node_map: Dict[str, Node]) -> str:
        path = []
        current = node
        visited = set() 
        while current and current.parent_id and current.parent_id != current.doc_id and current.id not in visited:
            visited.add(current.id)
            parent = node_map.get(current.parent_id)
            if not parent:
                break
            if parent.modality_category == ModalityCategory.DOCUMENT_STRUCTURE and parent.modality in ['title', 'heading']:
                clean_title = parent.content.split('[SUMMARY]')[0].split('\n')[0].strip()
                if clean_title:
                    path.append(clean_title)
            current = parent
        path.reverse()
        return " > ".join(path)

    def _resolve_cross_references(self, nodes: List[Node], edges: List[Edge], ref_lookup: Dict[str, Dict[str, str]]):
        existing_edges = {(e.source_id, e.target_id, e.type) for e in edges}
        for node in nodes:
            if node.modality_category != ModalityCategory.TEXTUAL_CONTENT:
                continue
            for match in self.REFERENCE_REGEX.finditer(node.content):
                ref_text = match.group(0).lower()
                ref_num = match.group(1)
                target_id = None
                target_modality = "element"
                if "fig" in ref_text:
                    target_id = ref_lookup["figure"].get(ref_num)
                    target_modality = "figure"
                elif "tab" in ref_text:
                    target_id = ref_lookup["table"].get(ref_num)
                    target_modality = "table"
                elif "eq" in ref_text:
                    target_id = ref_lookup["formula"].get(ref_num)
                    target_modality = "equation"
                elif "sec" in ref_text:
                    target_id = ref_lookup["section"].get(ref_num)
                    target_modality = "section"
                if target_id and target_id != node.id:
                    edge_key = (node.id, target_id, "references")
                    if edge_key not in existing_edges:
                        edges.append(Edge(
                            source_id=node.id, target_id=target_id, type="references",
                            type_category=EdgeCategory.REFERENCE, creator_processor="post_processor",
                            edge_meta={"matched_text": match.group(0), "target_type": target_modality}
                        ))
                        existing_edges.add(edge_key)
            for match in self.PAGE_REFERENCE_REGEX.finditer(node.content):
                ref_num = match.group(1)
                target_id = ref_lookup["page"].get(ref_num)
                if target_id and target_id != node.id:
                    edge_key = (node.id, target_id, "references")
                    if edge_key not in existing_edges:
                        edges.append(Edge(
                            source_id=node.id, target_id=target_id, type="references",
                            type_category=EdgeCategory.REFERENCE, creator_processor="post_processor",
                            edge_meta={"matched_text": match.group(0), "target_type": "page"}
                        ))
                        existing_edges.add(edge_key)

    def _extract_spatial_relations(self, nodes: List[Node], edges: List[Edge]):
        page_groups: Dict[int, List[Node]] = {}
        for node in nodes:
            if node.bbox and node.modality_category in self.SPATIAL_CATEGORIES:
                p = node.bbox.page
                if p not in page_groups:
                    page_groups[p] = []
                page_groups[p].append(node)
        existing_edges = {(e.source_id, e.target_id, e.type) for e in edges}
        for page, page_nodes in page_groups.items():
            boxes = []
            for n in page_nodes:
                b = n.bbox
                poly = box(b.x, b.y, b.x + b.w, b.y + b.h)
                boxes.append((n, poly))
            for i in range(len(boxes)):
                n1, b1 = boxes[i]
                for j in range(i + 1, len(boxes)):
                    n2, b2 = boxes[j]
                    if b1.distance(b2) < self.adjacency_threshold:
                        self._add_spatial_edge(edges, existing_edges, n1, n2, "adjacent_to", bidirectional=True)
                    y_overlap = self._get_overlap(b1.bounds[1], b1.bounds[3], b2.bounds[1], b2.bounds[3])
                    min_height = min(b1.bounds[3] - b1.bounds[1], b2.bounds[3] - b2.bounds[1])
                    if min_height > 0 and (y_overlap / min_height) >= self.band_overlap:
                        if b1.bounds[2] <= b2.bounds[0]: 
                            self._add_spatial_edge(edges, existing_edges, n1, n2, "left_of", bidirectional=False)
                            self._add_spatial_edge(edges, existing_edges, n2, n1, "right_of", bidirectional=False)
                    x_overlap = self._get_overlap(b1.bounds[0], b1.bounds[2], b2.bounds[0], b2.bounds[2])
                    min_width = min(b1.bounds[2] - b1.bounds[0], b2.bounds[2] - b2.bounds[0])
                    if min_width > 0 and (x_overlap / min_width) >= self.band_overlap:
                        if b1.bounds[3] <= b2.bounds[1]: 
                            self._add_spatial_edge(edges, existing_edges, n1, n2, "above", bidirectional=False)
                            self._add_spatial_edge(edges, existing_edges, n2, n1, "below", bidirectional=False)

    def _get_overlap(self, min1: float, max1: float, min2: float, max2: float) -> float:
        return max(0, min(max1, max2) - max(min1, min2))

    def _add_spatial_edge(self, edges: List[Edge], existing: set, src: Node, tgt: Node, rel_type: str, bidirectional: bool):
        edge_key = (src.id, tgt.id, rel_type)
        if edge_key in existing:
            return
        edges.append(Edge(
            source_id=src.id, target_id=tgt.id, type=rel_type,
            type_category=EdgeCategory.SPATIAL_RELATION, is_bidirectional=bidirectional,
            creator_processor="post_processor"
        ))
        existing.add(edge_key)

    def _generate_edge_evidence(self, edges: List[Edge], node_map: Dict[str, Node]):
        for edge in edges:
            src = node_map.get(edge.source_id)
            tgt = node_map.get(edge.target_id)
            src_mod = src.modality if src else "Element"
            tgt_mod = tgt.modality if tgt else "Element"
            page_no = "Unknown"
            if src and src.bbox:
                page_no = src.bbox.page + 1 
            src_snip = self._get_snippet(src)
            tgt_snip = self._get_snippet(tgt)
            if edge.type in ["contains", "contains_section", "contains_page"]:
                edge.evidence = f"'{src_snip}' ({src_mod}) contains '{tgt_snip}' ({tgt_mod})."
            elif edge.type == "read_order_next":
                edge.evidence = f"'{tgt_snip}' ({tgt_mod}) follows '{src_snip}' ({src_mod}) in reading order."
            elif edge.type == "adjacent_to":
                edge.evidence = f"'{src_snip}' ({src_mod}) is spatially adjacent to '{tgt_snip}' ({tgt_mod}) on page {page_no}."
            elif edge.type == "left_of":
                edge.evidence = f"'{src_snip}' ({src_mod}) is positioned to the left of '{tgt_snip}' ({tgt_mod}) on page {page_no}."
            elif edge.type == "right_of":
                edge.evidence = f"'{src_snip}' ({src_mod}) is positioned to the right of '{tgt_snip}' ({tgt_mod}) on page {page_no}."
            elif edge.type == "above":
                edge.evidence = f"'{src_snip}' ({src_mod}) is positioned above '{tgt_snip}' ({tgt_mod}) on page {page_no}."
            elif edge.type == "below":
                edge.evidence = f"'{src_snip}' ({src_mod}) is positioned below '{tgt_snip}' ({tgt_mod}) on page {page_no}."
            elif edge.type == "references":
                meta = getattr(edge, 'edge_meta', {}) or {}
                matched_text = meta.get("matched_text", tgt_mod)
                edge.evidence = f"'{src_snip}' ({src_mod}) references {matched_text}."
            elif edge.type == "captioned_by":
                edge.evidence = f"'{src_snip}' ({src_mod}) is described by caption '{tgt_snip}'."
            else:
                edge.evidence = f"Relationship '{edge.type}' from '{src_snip}' to '{tgt_snip}'."

    def _get_snippet(self, node: Optional[Node], length: int = 40) -> str:
        if not node:
            return "Unknown"
        if node.modality == "page":
            return node.content if node.content else "Page"
        if node.content:
            clean = node.content.replace("[SECTION_CONTEXT]", "").replace("[CAPTION]:", "").strip()
            clean = re.sub(r'\|', ' ', clean)
            clean = re.sub(r'-{3,}', '', clean)
            clean = re.sub(r'\s+', ' ', clean).strip()
            return clean[:length] + "..." if len(clean) > length else clean
        return node.modality.capitalize()