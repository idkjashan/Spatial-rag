import sys
import os
import json
import asyncio
from dotenv import load_dotenv
load_dotenv()
from collections import defaultdict
from typing import List, Tuple, Dict

from core.parsers.docling_parser import DoclingParser
from core.processors.post_processor import GraphPostProcessor
from core.processors.contextual_enricher import ContextualEnricher
from core.embeddings.graph_embedder import GraphEmbedder  # FIXED: Updated import path
from core.models.node import Node, ModalityCategory
from core.models.edge import Edge, EdgeCategory
from core.models.document import Document

# Optional: for pretty printing
try:
    from rich import print
    from rich.table import Table
    from rich.tree import Tree
    from rich.console import Console
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

STATE_FILE = "enriched_output_full.json"

# ==========================================
# State Management (Save/Load)
# ==========================================

def save_state(document: Document, nodes: List[Node], edges: List[Edge]):
    """Saves the current pipeline state to JSON, stripping heavy embedding vectors."""
    nodes_clean = []
    for n in nodes:
        n_dict = n.model_dump()
        n_dict.get("node_meta", {}).pop("__embedding_vector__", None)
        nodes_clean.append(n_dict)
        
    edges_clean = []
    for e in edges:
        e_dict = e.model_dump()
        e_dict.get("edge_meta", {}).pop("__embedding_vector__", None)
        edges_clean.append(e_dict)
        
    output_json = {
        "document": document.model_dump(),
        "nodes": nodes_clean,
        "edges": edges_clean
    }
    with open(STATE_FILE, "w") as f:
        json.dump(output_json, f, indent=2, default=str)
    print(f"\n💾 State saved to {STATE_FILE}")

def load_state() -> Tuple[Document, List[Node], List[Edge]]:
    """Loads pipeline state from JSON."""
    if not os.path.exists(STATE_FILE):
        print(f"❌ State file {STATE_FILE} not found! You must run Step 1 first.")
        sys.exit(1)
        
    with open(STATE_FILE, "r") as f:
        data = json.load(f)
        
    document = Document(**data["document"])
    nodes = [Node(**n) for n in data["nodes"]]
    edges = [Edge(**e) for e in data["edges"]]
    
    print(f"📂 Loaded state from {STATE_FILE} (Doc: {document.id}, Nodes: {len(nodes)}, Edges: {len(edges)})")
    return document, nodes, edges

# ==========================================
# Interactive Menu
# ==========================================

def get_user_selection():
    print("\n🛠️ SpatialRAG Pipeline Test Menu")
    print("1. Parse PDF (Docling)")
    print("2. Post-Process (Spatial, Hierarchy, Context)")
    print("3. LLM Enrichment (VLM/SLM Summaries)")
    print("4. Graph Embedding (Dense Vectors)")
    print("Type 'all' to run 1 -> 2 -> 3 -> 4 sequentially.")
    
    selection = input("Enter steps to run (e.g., '1,2' or '3' or 'all'): ").strip().lower()
    
    if selection == 'all':
        return ['1', '2', '3', '4']
    return [s.strip() for s in selection.split(',') if s.strip()]

# ==========================================
# Printing & Verification Helpers
# ==========================================

def build_node_map(nodes: List[Node]) -> Dict[str, Node]:
    return {n.id: n for n in nodes}

def build_adjacency(edges: List[Edge], direction='outgoing', category_filter=None) -> Dict[str, List[str]]:
    adj = defaultdict(list)
    for e in edges:
        if category_filter and e.type_category != category_filter:
            continue
        if direction == 'outgoing':
            adj[e.source_id].append(e.target_id)
        else:
            adj[e.target_id].append(e.source_id)
    return adj

def print_hierarchy(nodes: List[Node], edges: List[Edge], doc_id: str, max_depth=5):
    node_map = build_node_map(nodes)
    adj = build_adjacency(edges, 'outgoing', category_filter=EdgeCategory.HIERARCHY)
    
    if doc_id not in node_map:
        print(f"Document node {doc_id} not found in nodes!")
        return
    
    def print_subtree(node_id: str, indent=0, depth=0):
        if depth > max_depth:
            print("  " * indent + "... (truncated)")
            return
        node = node_map.get(node_id)
        if not node: return
        label = f"{node.modality}"
        content_preview = node.content[:50].replace('\n', ' ') if node.content else ''
        if content_preview:
            label += f": {content_preview}..."
        print("  " * indent + f"└── {label} (id: {node.id[:8]})")
        for child_id in adj.get(node_id, []):
            print_subtree(child_id, indent + 1, depth + 1)
    
    print("\n📁 Document Hierarchy (Hierarchy Edges Only):")
    print_subtree(doc_id)

def print_statistics(nodes: List[Node], edges: List[Edge]):
    print(f"\n📊 Total Statistics:")
    print(f"   Total Nodes: {len(nodes)}")
    print(f"   Total Edges: {len(edges)}")
    
    mod_counts = defaultdict(int)
    for n in nodes: mod_counts[n.modality] += 1
    print("   Nodes by modality:")
    for mod, cnt in sorted(mod_counts.items(), key=lambda x: -x[1]):
        print(f"      {mod}: {cnt}")
    
    edge_counts = defaultdict(int)
    for e in edges: edge_counts[e.type] += 1
    print("   Edges by type:")
    for typ, cnt in sorted(edge_counts.items(), key=lambda x: -x[1]):
        print(f"      {typ}: {cnt}")

def print_post_processing_stats(nodes: List[Node], edges: List[Edge], node_map: Dict[str, Node]):
    print("\n🛠️ Post-Processing Verification:")
    augmented_count = sum(1 for n in nodes if ">" in n.content and n.modality_category in [ModalityCategory.TEXTUAL_CONTENT, ModalityCategory.TABLE_CONTAINER])
    print(f"   Nodes with Context Path Prefix: {augmented_count}")

    spatial_edges = [e for e in edges if e.type_category == EdgeCategory.SPATIAL_RELATION]
    print(f"   Spatial Edges Created: {len(spatial_edges)}")

    ref_edges = [e for e in edges if e.type_category == EdgeCategory.REFERENCE]
    print(f"   Reference Edges Created: {len(ref_edges)}")

def print_llm_enrichment_stats(nodes: List[Node]):
    print("\n🧠 Phase 2 Contextual Enrichment Verification:")
    vlm_count = sum(1 for n in nodes if "[VLM_SUMMARY]" in n.content)
    slm_table_count = sum(1 for n in nodes if "[SLM_SUMMARY]" in n.content and n.modality_category == ModalityCategory.TABLE_CONTAINER)
    slm_formula_count = sum(1 for n in nodes if "[SLM_EXPLANATION]" in n.content)
    
    print(f"   VLM Image Summaries Added: {vlm_count}")
    print(f"   SLM Table Summaries Added: {slm_table_count}")
    print(f"   SLM Formula Explanations Added: {slm_formula_count}")

def print_embedding_stats(nodes: List[Node], edges: List[Edge], node_vectors: Dict[str, List[float]], edge_vectors: Dict[str, List[float]]):
    print("\n🧮 Phase 2.5 Graph Embedding Verification:")
    print(f"   Total Node Vectors Generated: {len(node_vectors)}")
    print(f"   Total Edge Vectors Generated: {len(edge_vectors)}")
    
    if node_vectors:
        sample_vec = list(node_vectors.values())[0]
        print(f"   Node Vector Dimensions: {len(sample_vec)}")
        
    nodes_with_vecs = sum(1 for n in nodes if n.node_meta.get("__embedding_vector__"))
    print(f"   Nodes with attached vector meta: {nodes_with_vecs}")

# ==========================================
# Main Pipeline Execution
# ==========================================

def run_pipeline(pdf_path: str, steps: List[str]):
    document, nodes, edges = None, None, None
    
    # STEP 1: Parse PDF
    if '1' in steps:
        print(f"\n🚀 Step 1: Parsing {pdf_path}...")
        parser = DoclingParser(tenant_id="test_tenant", image_cache_path="./images_cache")
        document, nodes, edges = parser.parse(pdf_path)
        save_state(document, nodes, edges)
    else:
        document, nodes, edges = load_state()
        
    # STEP 2: Post-Process
    if '2' in steps:
        print("\n🛠️ Step 2: Running GraphPostProcessor...")
        post_processor = GraphPostProcessor()
        document, nodes, edges = post_processor.process(document, nodes, edges)
        print("✅ Post-Processing Complete.")
        save_state(document, nodes, edges)
        print_post_processing_stats(nodes, edges, build_node_map(nodes))
        
    # STEP 3: LLM Enrichment
    if '3' in steps:
        print("\n🧠 Step 3: Running ContextualEnricher (VLM/SLM)...")
        
        enricher = ContextualEnricher(
            llm_base_url="https://api.groq.com/openai/v1",
            llm_api_key=os.getenv("GROQ_API_KEY"),
            slm_model="openai/gpt-oss-20b",
            vlm_model="meta-llama/llama-4-scout-17b-16e-instruct",
            max_concurrent_requests=5
        )
        document, nodes, edges = enricher.process(document, nodes, edges)
        print("✅ Contextual Enrichment Complete.")
        save_state(document, nodes, edges)
        print_llm_enrichment_stats(nodes)
        
    # STEP 4: Graph Embedding
    if '4' in steps:
        print("\n🧮 Step 4: Running GraphEmbedder (Pillar A & C)...")
        
        embedder = GraphEmbedder(
            base_url="https://api.jina.ai/v1",
            api_key=os.getenv("JINA_API_KEY"),
            model_name="jina-embeddings-v4",
            batch_size=64
        )
        document, nodes, edges, node_vectors, edge_vectors = embedder.process(document, nodes, edges)
        print("✅ Graph Embedding Complete.")
        save_state(document, nodes, edges)
        print_embedding_stats(nodes, edges, node_vectors, edge_vectors)

    # Final Summary Print
    print_statistics(nodes, edges)
    print_hierarchy(nodes, edges, document.id)

if __name__ == "__main__":
    pdf_file = "./CRNN.pdf"
    if not os.path.exists(pdf_file):
        print(f"❌ PDF not found at {pdf_file}")
        sys.exit(1)
        
    selected_steps = get_user_selection()
    run_pipeline(pdf_file, selected_steps)