import sys
import os
import json
from pathlib import Path
from collections import defaultdict

from core.parsers.docling_parser import DoclingParser
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

def build_adjacency(edges, direction='outgoing'):
    """Build adjacency list: parent -> list of children"""
    adj = defaultdict(list)
    for e in edges:
        if direction == 'outgoing':
            adj[e.source_id].append(e.target_id)
        else:
            adj[e.target_id].append(e.source_id)
    return adj

def print_hierarchy(nodes, edges, doc_id, max_depth=5):
    """Print document hierarchy as a tree."""
    node_map = build_node_map(nodes)
    adj = build_adjacency(edges, 'outgoing')
    
    # Find root: document node (doc_id)
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
        # Get a label
        label = f"{node.modality}"
        content_preview = node.content[:50].replace('\n', ' ') if node.content else ''
        if content_preview:
            label += f": {content_preview}..."
        print("  " * indent + f"└── {label} (id: {node.id[:8]})")
        for child_id in adj.get(node_id, []):
            print_subtree(child_id, indent + 1, depth + 1)
    
    print("\n📁 Document Hierarchy:")
    print_subtree(root_id)

def print_sample_nodes(nodes, count=10):
    print(f"\n🔹 Sample Nodes (first {count}):")
    for i, node in enumerate(nodes[:count]):
        content = node.content[:80].replace('\n', ' ') if node.content else '(empty)'
        bbox = f"({node.bbox.x:.0f},{node.bbox.y:.0f}) {node.bbox.w:.0f}x{node.bbox.h:.0f}" if node.bbox else 'None'
        print(f"  [{i}] {node.modality} [{node.modality_category.name}] - content: {content}...")
        print(f"       bbox: {bbox}, page: {node.bbox.page if node.bbox else '?'}")

def print_sample_edges(edges, node_map, count=10):
    print(f"\n🔹 Sample Edges (first {count}):")
    for i, edge in enumerate(edges[:count]):
        src = node_map.get(edge.source_id)
        tgt = node_map.get(edge.target_id)
        src_label = src.modality if src else '?'
        tgt_label = tgt.modality if tgt else '?'
        print(f"  [{i}] {edge.type} ({edge.type_category.name}) : {src_label}({edge.source_id[:8]}) -> {tgt_label}({edge.target_id[:8]})")

def print_statistics(nodes, edges):
    print(f"\n📊 Statistics:")
    print(f"   Total Nodes: {len(nodes)}")
    print(f"   Total Edges: {len(edges)}")
    
    # Count by modality
    mod_counts = defaultdict(int)
    for n in nodes:
        mod_counts[n.modality] += 1
    print("   Nodes by modality:")
    for mod, cnt in sorted(mod_counts.items(), key=lambda x: -x[1]):
        print(f"      {mod}: {cnt}")
    
    # Count by edge type
    edge_counts = defaultdict(int)
    for e in edges:
        edge_counts[e.type] += 1
    print("   Edges by type:")
    for typ, cnt in sorted(edge_counts.items(), key=lambda x: -x[1]):
        print(f"      {typ}: {cnt}")

def test_parser(pdf_path):
    parser = DoclingParser(
        tenant_id="test_tenant",
        image_cache_path="./images_cache"
    )
    
    print(f"🚀 Parsing: {pdf_path}")
    document, nodes, edges = parser.parse(pdf_path)
    
    print(f"\n📄 Document: {document.id}")
    print(f"   Status: {document.status}")
    print(f"   Pages: {document.total_pages}")
    
    # Build maps
    node_map = build_node_map(nodes)
    
    # Print stats
    print_statistics(nodes, edges)
    
    # Sample nodes
    print_sample_nodes(nodes, count=10)
    print_sample_edges(edges, node_map, count=10)
    
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
    if missing:
        print("   Missing image paths:")
        for img in missing[:5]:
            print(f"      {img.image_path}")
    
    # Optional: Save full data to JSON for external inspection
    output_json = {
        "document": document.model_dump(),
        "nodes": [n.model_dump() for n in nodes],
        "edges": [e.model_dump() for e in edges]
    }
    with open("parser_output_full.json", "w") as f:
        json.dump(output_json, f, indent=2, default=str)
    print("\n📊 Full output saved to parser_output_full.json")
    
    return document, nodes, edges

if __name__ == "__main__":
    pdf_file = "./CRNN.pdf"
    if not os.path.exists(pdf_file):
        print(f"❌ PDF not found at {pdf_file}")
        sys.exit(1)
    test_parser(pdf_file)