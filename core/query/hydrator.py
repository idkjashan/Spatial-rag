import logging
from typing import List, Dict, Optional, Any, Tuple
from collections import defaultdict

from psycopg2.extras import RealDictCursor

from core.models.node import Node, ModalityCategory
from core.models.edge import Edge
from core.stores.postgres_store import PostgresStore

logger = logging.getLogger(__name__)

class ContextHydrator:
    """
    Phase 4.3: Hydrates Node and Edge UUIDs from Postgres and assembles
    a structured, LLM-ready context string.
    """
    
    def __init__(self, pg_store: PostgresStore):
        self.pg_store = pg_store

    async def hydrate(self, retrieval_result: Dict[str, List[str]]) -> str:
        """
        Main execution method. Fetches data from Postgres and formats it.
        """
        node_ids = retrieval_result.get("node_ids", [])
        edge_ids = retrieval_result.get("edge_ids", [])
        
        if not node_ids:
            return "No relevant context was found in the database for this query."

        # 1. Fetch full data from Postgres
        nodes = await self._fetch_nodes(node_ids)
        edges = await self._fetch_edges(edge_ids)
        
        # 2. Group and sort nodes to reconstruct document hierarchy
        grouped_context = self._group_and_sort_nodes(nodes)
        
        # 3. Format the final LLM prompt string
        formatted_string = self._format_context(grouped_context, edges)
        
        logger.info(f"Hydrated context: {len(nodes)} nodes, {len(edges)} edges. Context length: {len(formatted_string)} chars.")
        return formatted_string

    # ==========================================
    # Database Fetching
    # ==========================================
    async def _fetch_nodes(self, node_ids: List[str]) -> List[Node]:
        if not node_ids: return []
        conn = self.pg_store._get_conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Use ANY for efficient array matching
                cur.execute("""
                    SELECT * FROM nodes 
                    WHERE id = ANY(%s::uuid[])
                """, (node_ids,))
                records = cur.fetchall()
                
                # Reconstruct Pydantic models (handling JSONB fields)
                nodes = []
                for rec in records:
                    try:
                        # Convert dict to Pydantic Node safely
                        nodes.append(Node(**{k: v for k, v in rec.items() if v is not None}))
                    except Exception as e:
                        logger.error(f"Failed to parse node {rec.get('id')}: {e}")
                return nodes
        finally:
            self.pg_store._put_conn(conn)

    async def _fetch_edges(self, edge_ids: List[str]) -> List[Edge]:
        if not edge_ids: return []
        conn = self.pg_store._get_conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM edges 
                    WHERE id = ANY(%s::uuid[])
                """, (edge_ids,))
                records = cur.fetchall()
                
                edges = []
                for rec in records:
                    try:
                        edges.append(Edge(**{k: v for k, v in rec.items() if v is not None}))
                    except Exception as e:
                        logger.error(f"Failed to parse edge {rec.get('id')}: {e}")
                return edges
        finally:
            self.pg_store._put_conn(conn)

    # ==========================================
    # Context Reconstruction Logic
    # ==========================================
    def _group_and_sort_nodes(self, nodes: List[Node]) -> Dict[Optional[str], List[Node]]:
        """
        Groups nodes by their parent_id to maintain document structure.
        Sorts them by sequence_index to restore reading order.
        """
        grouped = defaultdict(list)
        for node in nodes:
            parent = node.parent_id if node.parent_id else "ORPHAN"
            grouped[parent].append(node)
            
        # Sort each group by sequence_index
        for parent_id in grouped.keys():
            grouped[parent_id].sort(key=lambda n: n.sequence_index)
            
        return grouped

    def _format_context(self, grouped_nodes: Dict[str, List[Node]], edges: List[Edge]) -> str:
        """Builds the final markdown-like string for the LLM."""
        context_blocks = []
        
        # 1. Format Nodes by Parent Group
        for parent_id, node_list in grouped_nodes.items():
            if parent_id != "ORPHAN":
                context_blocks.append(f"[Context Group from Parent ID: {parent_id}]")
            else:
                context_blocks.append(f"[Standalone Context]")
                
            for node in node_list:
                block = self._format_node(node)
                if block:
                    context_blocks.append(block)
                    
        # 2. Format Edges (Relationships)
        if edges:
            context_blocks.append("\n--- OBSERVED RELATIONSHIPS (Graph Edges) ---")
            node_lookup = {str(n.id): n for n in [item for sublist in grouped_nodes.values() for item in sublist]}
            
            for edge in edges:
                src = node_lookup.get(str(edge.source_id))
                tgt = node_lookup.get(str(edge.target_id))
                
                src_name = src.modality if src else "Unknown Node"
                tgt_name = tgt.modality if tgt else "Unknown Node"
                
                # Use edge evidence if available, otherwise construct from type
                rel_desc = edge.evidence or f"{src_name} --{edge.type}--> {tgt_name}"
                context_blocks.append(f"- {rel_desc}")
                
        return "\n".join(context_blocks)

    def _format_node(self, node: Node) -> str:
        """Formats a single node into a readable string block based on its modality."""
        modality = node.modality_category.value
        
        # Clean content of internal tags
        content = node.content.replace("[SECTION_CONTEXT]", "").replace("[CAPTION]:", "Caption:").strip()
        
        if modality == ModalityCategory.TEXTUAL_CONTENT.value:
            return f"- [Textual Content | Seq: {node.sequence_index}]: {content}"
            
        elif modality == ModalityCategory.DOCUMENT_STRUCTURE.value:
            return f"- [Document Structure | Seq: {node.sequence_index}]: {content}"
            
        elif modality == ModalityCategory.TABLE_CONTAINER.value:
            return f"- [Table Container | Seq: {node.sequence_index}]:\n{content}"
            
        elif modality == ModalityCategory.TABLE_CONTENT.value:
            # Extract row data from meta if available
            row_data = node.node_meta
            return f"- [Table Row | Seq: {node.sequence_index}]: {content} (Data: {row_data})"
            
        elif modality == ModalityCategory.IMAGE.value:
            # Include image path for multimodal LLMs to process
            return f"- [Image | Seq: {node.sequence_index}]: Path: {node.image_path} | Description: {content}"
            
        elif modality == ModalityCategory.EQUATION.value:
            return f"- [Equation | Seq: {node.sequence_index}]: {content}"
            
        else:
            return f"- [{modality} | Seq: {node.sequence_index}]: {content}"
