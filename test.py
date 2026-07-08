import sys
import os
import json
from pathlib import Path
from collections import defaultdict

from core.parsers.docling_parser import DoclingParser
from core.processors.post_processor import GraphPostProcessor  # Added Post-Processor
from core.models.node import Node, ModalityCategory
from core.models.edge import Edge, EdgeCategory

# Optional: for pretty printing
try:
    from rich import print
    from rich.table import Table
    from rich.tree import Tree
    from rich.console import Console
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

def build_node_map(nodes):
    return {n.id: n for n in nodes}

def build_adjacency(edges, direction='outgoing', category_filter=None):
    """Build adjacency list: parent -> list of children"""
    adj = defaultdict(list)
    for e in edges:
        if category_filter and e.type_category != category_filter:
            continue
        if direction == 'outgoing':
            adj[e.source_id].append(e.target_id)
        else:
            adj[e.target_id].append(e.source_id)
    return adj

def print_hierarchy(nodes, edges, doc_id, max_depth=5):
    """Print document hierarchy as a tree (Strictly HIERARCHY edges only)."""
    node_map = build_node_map(nodes)
    # Filter for HIERARCHY only so spatial/reference edges don't ruin the tree
    adj = build_adjacency(edges, 'outgoing', category_filter=EdgeCategory.HIERARCHY)
    
    root_id = doc_id
    if root_id not in node_map:
        print(f"Document node {root_id} not found in nodes!")
        return
    
    def print_subtree(node_id, indent=0, depth=0):
        if depth > max_depth:
            print("  " * indent + "... (truncated)")
            return
        node = node_map.get(node_id)
        if not node:
            return
        label = f"{node.modality}"
        content_preview = node.content[:50].replace('\n', ' ') if node.content else ''
        if content_preview:
            label += f": {content_preview}..."
        print("  " * indent + f"└── {label} (id: {node.id[:8]})")
        for child_id in adj.get(node_id, []):
            print_subtree(child_id, indent + 1, depth + 1)
    
    print("\n📁 Document Hierarchy (Hierarchy Edges Only):")
    print_subtree(root_id)

def print_sample_nodes(nodes, count=10):
    print(f"\n🔹 Sample Nodes (first {count}):")
    for i, node in enumerate(nodes[:count]):
        content = node.content[:80].replace('\n', ' ') if node.content else '(empty)'
        bbox = f"({node.bbox.x:.0f},{node.bbox.y:.0f}) {node.bbox.w:.0f}x{node.bbox.h:.0f}" if node.bbox else 'None'
        print(f"  [{i}] {node.modality} [{node.modality_category.name}] - content: {content}...")
        print(f"       bbox: {bbox}, page: {node.bbox.page if node.bbox else '?'}")

def print_post_processing_stats(nodes, edges, node_map):
    """Prints stats specifically for Phase 0.75 enrichments."""
    print("\n🛠️ Post-Processing Verification:")
    
    # 1. Context Augmentation Check
    augmented_count = sum(1 for n in nodes if ">" in n.content and n.modality_category in [ModalityCategory.TEXTUAL_CONTENT, ModalityCategory.TABLE_CONTAINER])
    print(f"   Nodes with Context Path Prefix: {augmented_count}")
    if augmented_count > 0:
        sample = next(n for n in nodes if ">" in n.content and n.modality_category == ModalityCategory.TEXTUAL_CONTENT)
        print(f"   Sample Context: {sample.content[:100]}...")

    # 2. Spatial Relations Check
    spatial_edges = [e for e in edges if e.type_category == EdgeCategory.SPATIAL_RELATION]
    print(f"\n   Spatial Edges Created: {len(spatial_edges)}")
    for e in spatial_edges[:3]:
        src = node_map.get(e.source_id)
        tgt = node_map.get(e.target_id)
        print(f"   - {src.modality} -> {e.type} -> {tgt.modality}")

    # 3. Cross-References Check
    ref_edges = [e for e in edges if e.type_category == EdgeCategory.REFERENCE]
    print(f"\n   Reference Edges Created: {len(ref_edges)}")
    for e in ref_edges[:3]:
        src = node_map.get(e.source_id)
        tgt = node_map.get(e.target_id)
        print(f"   - Text({src.id[:8]}) -> references -> {tgt.modality} (Matched: {e.edge_meta.get('matched_text', 'N/A')})")

    # 4. Evidence Generation Check
    evidenced_edges = [e for e in edges if e.evidence]
    print(f"\n   Edges with Evidence String: {len(evidenced_edges)} / {len(edges)}")
    if evidenced_edges:
        print("   Sample Evidence Strings:")
        for e in evidenced_edges[:3]:
            print(f"   - [{e.type}]: {e.evidence}")

def print_statistics(nodes, edges):
    print(f"\n📊 Total Statistics:")
    print(f"   Total Nodes: {len(nodes)}")
    print(f"   Total Edges: {len(edges)}")
    
    mod_counts = defaultdict(int)
    for n in nodes:
        mod_counts[n.modality] += 1
    print("   Nodes by modality:")
    for mod, cnt in sorted(mod_counts.items(), key=lambda x: -x[1]):
        print(f"      {mod}: {cnt}")
    
    edge_counts = defaultdict(int)
    for e in edges:
        edge_counts[e.type] += 1
    print("   Edges by type:")
    for typ, cnt in sorted(edge_counts.items(), key=lambda x: -x[1]):
        print(f"      {typ}: {cnt}")

def test_pipeline(pdf_path):
    # 1. Initialize Parser
    parser = DoclingParser(
        tenant_id="test_tenant",
        image_cache_path="./images_cache"
    )
    
    print(f"🚀 Parsing: {pdf_path}")
    document, nodes, edges = parser.parse(pdf_path)
    
    print(f"\n📄 Document Parsed: {document.id}")
    print(f"   Status: {document.status}")
    print(f"   Pages: {document.total_pages}")
    
    # 2. Initialize and Run Post-Processor
    print("\n🛠️ Running GraphPostProcessor...")
    post_processor = GraphPostProcessor()
    document, nodes, edges = post_processor.process(document, nodes, edges)
    print("✅ Post-Processing Complete.")
    
    # Build maps for printing
    node_map = build_node_map(nodes)
    
    # 3. Print Stats and Verifications
    print_statistics(nodes, edges)
    print_post_processing_stats(nodes, edges, node_map)
    
    # Sample nodes
    print_sample_nodes(nodes, count=5)
    
    # Hierarchy tree
    print_hierarchy(nodes, edges, document.id)
    
    # Check images
    image_nodes = [n for n in nodes if n.modality_category == ModalityCategory.IMAGE]
    print(f"\n🖼️ {len(image_nodes)} image nodes.")
    missing = []
    saved = []
    for img in image_nodes:
        if os.path.exists(img.image_path):
            saved.append(img)
        else:
            missing.append(img)
    print(f"   Saved: {len(saved)}")
    print(f"   Missing: {len(missing)}")
    
    # Save full enriched data to JSON
    output_json = {
        "document": document.model_dump(),
        "nodes": [n.model_dump() for n in nodes],
        "edges": [e.model_dump() for e in edges]
    }
    with open("enriched_output_full.json", "w") as f:
        json.dump(output_json, f, indent=2, default=str)
    print("\n📊 Full enriched output saved to enriched_output_full.json")
    
    return document, nodes, edges

if __name__ == "__main__":
    pdf_file = "./CRNN.pdf"
    if not os.path.exists(pdf_file):
        print(f"❌ PDF not found at {pdf_file}")
        sys.exit(1)
    test_pipeline(pdf_file)