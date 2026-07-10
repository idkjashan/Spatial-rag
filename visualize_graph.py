import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Circle
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple

try:
    import mplcursors
    MPLCURSORS_AVAILABLE = True
except ImportError:
    MPLCURSORS_AVAILABLE = False
    print("⚠️ mplcursors not installed. Install with: pip install mplcursors")

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    print("⚠️ networkx not installed. Subgraph views require networkx.\nInstall: pip install networkx")

class DoclingGraphVisualizer:
    NODE_COLORS = {
        "page": "lightgrey",
        "title": "gold",
        "heading": "orange",
        "page_header": "lightblue",
        "page_footer": "lightblue",
        "paragraph": "lightgreen",
        "list_item": "lightgreen",
        "caption": "salmon",
        "formula": "violet",
        "figure": "cyan",
        "table": "pink",
        "text": "lightgreen",
        "equation": "mediumpurple",
        "list": "lightcoral",
        "table_row": "lightpink",
        "unknown": "white"
    }

    EDGE_COLORS = {
        "contains_page": "black",
        "contains_section": "brown",
        "contains": "grey",
        "captioned_by": "red",
        "read_order_next": "blue",
        "contains_row": "plum",
        "contains_item": "dimgrey",
        "contains_latex": "orange",
        "adjacent_to": "green",
        "left_of": "yellow",
        "right_of": "yellow",
        "above": "yellow",
        "below": "yellow",
        "references": "pink",
        "default": "grey"
    }

    def __init__(self, json_path: Optional[str] = None,
                 nodes: Optional[List[Dict]] = None,
                 edges: Optional[List[Dict]] = None):
        if json_path:
            with open(json_path, 'r') as f:
                data = json.load(f)
            self.nodes = data['nodes']
            self.edges = data['edges']
            self.doc = data.get('document', {})
            self.json_path = json_path
        else:
            self.nodes = nodes or []
            self.edges = edges or []
            self.doc = {}
            self.json_path = None

        self.node_map = {n['id']: n for n in self.nodes}
        self.edge_map = {e['id']: e for e in self.edges if 'id' in e}

        # Page grouping (nodes with bbox)
        self.page_groups = defaultdict(list)
        for n in self.nodes:
            if 'bbox' in n and n['bbox'] is not None:
                self.page_groups[n['bbox']['page']].append(n)

        self.page_dims = {}
        for n in self.nodes:
            if n.get('modality') == 'page' and n.get('node_meta'):
                if 'bbox' in n and n['bbox'] is not None:
                    p = n['bbox']['page']
                    self.page_dims[p] = (n['node_meta']['width'], n['node_meta']['height'])

        # Cross‑page connections
        self.node_cross_connections = defaultdict(set)
        for edge in self.edges:
            src = self.node_map.get(edge['source_id'])
            tgt = self.node_map.get(edge['target_id'])
            if not src or not tgt:
                continue
            if src.get('bbox') is None or tgt.get('bbox') is None:
                continue
            if src['bbox']['page'] != tgt['bbox']['page']:
                self.node_cross_connections[edge['source_id']].add(tgt['bbox']['page'])
                self.node_cross_connections[edge['target_id']].add(src['bbox']['page'])

        # Subgraph groups
        self.subgraph_groups = defaultdict(list)
        for n in self.nodes:
            if n.get('subgraph_id'):
                self.subgraph_groups[n['subgraph_id']].append(n['id'])

        # Map subgraph IDs to numbers for easy selection
        self.subgraph_index = {i+1: sid for i, sid in enumerate(sorted(self.subgraph_groups.keys()))}

    def _get_node_color(self, node):
        return self.NODE_COLORS.get(node.get('modality', 'unknown'), 'white')

    def _get_edge_color(self, edge):
        return self.EDGE_COLORS.get(edge.get('type', 'default'), self.EDGE_COLORS['default'])

    # ---------- Page‑based drawing ----------
    def _draw_node(self, ax, node, page_width, page_height, show_cross_badge=True):
        bbox = node['bbox']
        x = bbox['x'] * page_width
        y = bbox['y'] * page_height
        w = bbox['w'] * page_width
        h = bbox['h'] * page_height

        rect = patches.Rectangle(
            (x, y), w, h,
            linewidth=1, edgecolor='black',
            facecolor=self._get_node_color(node),
            alpha=0.6,
            gid=node['id'],
            picker=True
        )
        ax.add_patch(rect)

        label = node.get('modality', '')
        if label and w > 20 and h > 10:
            ax.text(x + w/2, y + h/2, label,
                    ha='center', va='center',
                    fontsize=8, color='black', weight='bold')

        if show_cross_badge and node['id'] in self.node_cross_connections:
            other_pages = sorted(self.node_cross_connections[node['id']])
            badge_text = f"↗ {','.join(str(p+1) for p in other_pages)}"
            ax.text(x + w, y - 5, badge_text,
                    ha='right', va='bottom',
                    fontsize=7, color='red', weight='bold',
                    bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.2', alpha=0.8))

    def _draw_edge(self, ax, edge, page_width, page_height, draw_arrow=False):
        src = self.node_map.get(edge['source_id'])
        tgt = self.node_map.get(edge['target_id'])
        if not src or not tgt:
            return
        if src.get('bbox') is None or tgt.get('bbox') is None:
            return

        sx = (src['bbox']['x'] + src['bbox']['w']/2) * page_width
        sy = (src['bbox']['y'] + src['bbox']['h']/2) * page_height
        tx = (tgt['bbox']['x'] + tgt['bbox']['w']/2) * page_width
        ty = (tgt['bbox']['y'] + tgt['bbox']['h']/2) * page_height

        color = self._get_edge_color(edge)
        edge_id = edge.get('id')
        line_kw = dict(color=color, linewidth=2.0, alpha=0.8, gid=edge_id)

        if draw_arrow and edge.get('type') == 'read_order_next':
            arrow = FancyArrowPatch((sx, sy), (tx, ty),
                                    arrowstyle='->', mutation_scale=15,
                                    **line_kw,
                                    picker=True)
            ax.add_patch(arrow)
            arrow.set_gid(edge_id)
            arrow.set_picker(5)
            ax.plot([sx, tx], [sy, ty],
                    color=color, linewidth=8, alpha=0.0,
                    gid=edge_id, picker=True)
        else:
            line, = ax.plot([sx, tx], [sy, ty], linestyle='-', picker=True, **line_kw)
            line.set_gid(edge_id)

        mx = (sx + tx) / 2
        my = (sy + ty) / 2
        circle = Circle((mx, my), radius=4,
                        facecolor=color, edgecolor='black', alpha=0.9,
                        gid=edge_id, picker=True)
        ax.add_patch(circle)

    def draw(self, page_num: Optional[int] = None,
             output_path: Optional[str] = None,
             dpi: int = 100,
             mode: str = 'interactive',
             show_read_order_arrows: bool = True,
             show_sequence_numbers: bool = False,
             show_cross_badge: bool = True):
        if not self.page_groups:
            print("No nodes with bbox found.")
            return

        if mode == 'interactive':
            if page_num is None:
                page_num = 0
            if page_num not in self.page_groups:
                print(f"Page {page_num} has no nodes.")
                return
            pages = {page_num: self.page_groups[page_num]}
            interactive = True
        else:
            if page_num is not None:
                print("mode='save_all' overrides page_num; drawing all pages.")
            pages = dict(sorted(self.page_groups.items()))
            interactive = False

        n_pages = len(pages)
        cols = min(3, n_pages)
        rows = (n_pages + cols - 1) // cols

        figsize = (6, 8) if n_pages == 1 else (cols*6, rows*8)
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if n_pages == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        node_handles, node_labels = [], []
        edge_handles, edge_labels = [], []
        artists_with_gid = []

        for idx, (p, node_list) in enumerate(pages.items()):
            ax = axes[idx]
            w, h = self.page_dims.get(p, (612, 792))
            ax.set_xlim(0, w)
            ax.set_ylim(0, h)
            ax.set_aspect('equal')
            ax.set_title(f"Page {p+1}  ({w:.0f}×{h:.0f})")
            ax.set_facecolor('#f5f5f5')

            for node in node_list:
                self._draw_node(ax, node, w, h, show_cross_badge)
                if show_sequence_numbers and 'sequence_index' in node:
                    bbox = node['bbox']
                    cx = (bbox['x'] + bbox['w']/2) * w
                    cy = (bbox['y'] + bbox['h']/2) * h
                    ax.text(cx, cy + 5, str(node['sequence_index']),
                            ha='center', va='bottom', fontsize=7,
                            color='red', weight='bold', bbox=dict(facecolor='white', alpha=0.7))

                col = self._get_node_color(node)
                label = node.get('modality', 'unknown')
                if (col, label) not in node_labels:
                    node_labels.append((col, label))
                    node_handles.append(
                        patches.Patch(facecolor=col, edgecolor='black', alpha=0.6, label=label)
                    )

            for edge in self.edges:
                src = self.node_map.get(edge['source_id'])
                tgt = self.node_map.get(edge['target_id'])
                if not src or not tgt:
                    continue
                if src.get('bbox') is None or tgt.get('bbox') is None:
                    continue
                if src['bbox']['page'] != p or tgt['bbox']['page'] != p:
                    continue
                self._draw_edge(ax, edge, w, h, draw_arrow=show_read_order_arrows)
                ecol = self._get_edge_color(edge)
                etype = edge.get('type', 'default')
                if (ecol, etype) not in edge_labels:
                    edge_labels.append((ecol, etype))
                    edge_handles.append(
                        Line2D([0], [0], color=ecol, linewidth=2, label=etype)
                    )

            for artist in ax.get_children():
                if hasattr(artist, 'get_gid') and artist.get_gid() is not None:
                    if hasattr(artist, 'get_picker') and artist.get_picker() is not None:
                        artists_with_gid.append(artist)
                    elif isinstance(artist, (Line2D, patches.Rectangle, Circle, FancyArrowPatch)):
                        artists_with_gid.append(artist)

        for ax in axes[len(pages):]:
            fig.delaxes(ax)

        if node_handles:
            fig.legend(handles=node_handles, loc='upper left',
                       title='Node Modality', bbox_to_anchor=(0.02, 0.98))
        if edge_handles:
            fig.legend(handles=edge_handles, loc='upper right',
                       title='Edge Type', bbox_to_anchor=(0.98, 0.98))

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        if interactive and MPLCURSORS_AVAILABLE and artists_with_gid:
            cursor = mplcursors.cursor(artists_with_gid, hover=False)
            @cursor.connect("add")
            def on_add(sel):
                artist = sel.artist
                gid = artist.get_gid()
                if gid is None:
                    return
                if gid in self.node_map:
                    node = self.node_map[gid]
                    content = node.get('content', '')
                    if len(content) > 300:
                        content = content[:300] + '…'
                    text = (
                        f"ID: {gid[:8]}\n"
                        f"Modality: {node.get('modality', 'unknown')}\n"
                        f"Page: {node['bbox']['page']+1}\n"
                        f"Seq: {node.get('sequence_index', 'N/A')}\n"
                        f"Content: {content}"
                    )
                elif gid in self.edge_map:
                    edge = self.edge_map[gid]
                    src = edge.get('source_id', '')[:8]
                    tgt = edge.get('target_id', '')[:8]
                    text = (
                        f"Edge ID: {gid[:8]}\n"
                        f"Type: {edge.get('type', 'unknown')}\n"
                        f"From: {src}\n"
                        f"To: {tgt}\n"
                        f"Meta: {edge.get('edge_meta', {})}"
                    )
                else:
                    text = f"Unknown element (gid={gid})"
                sel.annotation.set_text(text)
                sel.annotation.get_bbox_patch().set(fc="white", alpha=0.9)
            self._cursor = cursor

        if output_path:
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
            print(f"✅ Visualization saved to {output_path}")
        else:
            plt.show()

    # ---------- Subgraph visualisation with smart layout ----------
    def _compute_subgraph_layout(self, subgraph_id: str) -> Dict[str, Tuple[float, float]]:
        """Custom layout for tables, lists, equations, otherwise spring."""
        node_ids = self.subgraph_groups[subgraph_id]
        sub_nodes = [self.node_map[nid] for nid in node_ids if nid in self.node_map]
        if not sub_nodes:
            return {}

        # Find container
        container = next((n for n in sub_nodes if n.get('subgraph_role') == 'container'), None)
        if not container:
            container = sub_nodes[0]  # fallback

        modality = container.get('modality', '')
        pos = {}

        if modality == 'table':
            # Arrange rows vertically, columns horizontally
            rows = [n for n in sub_nodes if n.get('modality') == 'table_row']
            rows.sort(key=lambda n: n.get('node_meta', {}).get('row_index', 0))
            if rows:
                nrows = len(rows)
                # Also include the container itself
                nodes_to_place = [container] + rows
                # Horizontal spacing: container on left, rows to the right
                x0, y0 = 0, 0
                spacing = 2.0
                # Place container at (0,0)
                pos[container['id']] = (0, 0)
                for i, row in enumerate(rows):
                    pos[row['id']] = (2.0, - (i+1) * 1.2)
                # Additional members (e.g., captions) get placed around
                others = [n for n in sub_nodes if n['id'] not in pos]
                for i, n in enumerate(others):
                    pos[n['id']] = (4.0 + i*1.5, 0)
            else:
                # Fallback to spring
                pos = self._spring_layout(subgraph_id)
        elif modality == 'list':
            # Arrange list items vertically
            items = [n for n in sub_nodes if n.get('modality') == 'list_item']
            items.sort(key=lambda n: n.get('sequence_index', 0))
            if items:
                pos[container['id']] = (0, 0)
                for i, item in enumerate(items):
                    pos[item['id']] = (0, - (i+1) * 1.2)
                others = [n for n in sub_nodes if n['id'] not in pos]
                for i, n in enumerate(others):
                    pos[n['id']] = (2.0 + i*1.5, 0)
            else:
                pos = self._spring_layout(subgraph_id)
        elif modality == 'equation':
            # Place container and formula child
            children = [n for n in sub_nodes if n.get('subgraph_role') == 'member']
            pos[container['id']] = (0, 0)
            for i, child in enumerate(children):
                pos[child['id']] = (1.5, -1.0 - i*1.0)
        else:
            pos = self._spring_layout(subgraph_id)

        return pos

    def _spring_layout(self, subgraph_id: str) -> Dict[str, Tuple[float, float]]:
        """Fallback spring layout using networkx."""
        if not NETWORKX_AVAILABLE:
            return {}
        node_ids = self.subgraph_groups[subgraph_id]
        G = nx.Graph()
        for nid in node_ids:
            G.add_node(nid)
        for edge in self.edges:
            if edge['source_id'] in node_ids and edge['target_id'] in node_ids:
                G.add_edge(edge['source_id'], edge['target_id'])
        if len(G.nodes) == 0:
            return {}
        try:
            pos = nx.spring_layout(G, seed=42, k=1.5)
        except:
            pos = {nid: (i*0.5, 0) for i, nid in enumerate(G.nodes)}
        return pos

    def draw_subgraph(self, subgraph_id: str,
                      output_path: Optional[str] = None,
                      dpi: int = 100,
                      interactive: bool = True):
        """Render a subgraph with smart layout."""
        if subgraph_id not in self.subgraph_groups:
            print(f"Subgraph {subgraph_id} not found.")
            return

        node_ids = self.subgraph_groups[subgraph_id]
        sub_nodes = [self.node_map[nid] for nid in node_ids if nid in self.node_map]
        if not sub_nodes:
            print("No nodes found.")
            return

        sub_edges = []
        for edge in self.edges:
            src = edge.get('source_id')
            tgt = edge.get('target_id')
            if src in node_ids and tgt in node_ids:
                sub_edges.append(edge)

        # Compute layout
        pos = self._compute_subgraph_layout(subgraph_id)
        if not pos:
            print("Failed to compute layout.")
            return

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_title(f"Subgraph {subgraph_id[:8]}  ({len(sub_nodes)} nodes, {len(sub_edges)} edges)")
        ax.set_axis_off()

        artists = []
        # Draw edges first (so nodes appear on top)
        for edge in sub_edges:
            src = edge['source_id']
            tgt = edge['target_id']
            if src not in pos or tgt not in pos:
                continue
            x1, y1 = pos[src]
            x2, y2 = pos[tgt]
            color = self._get_edge_color(edge)
            line, = ax.plot([x1, x2], [y1, y2], color=color, linewidth=2, alpha=0.7,
                            gid=edge.get('id'), picker=True)
            artists.append(line)

        # Draw nodes
        for nid, (x, y) in pos.items():
            node = self.node_map[nid]
            color = self._get_node_color(node)
            # Determine node size: container bigger
            size = 800 if node.get('subgraph_role') == 'container' else 500
            sc = ax.scatter(x, y, s=size, facecolor=color, edgecolor='black',
                            alpha=0.8, gid=nid, picker=True)
            artists.append(sc)

            # Node label: modality + short content
            modality = node.get('modality', 'unknown')
            content = node.get('content', '')
            # For table rows, show row data
            if modality == 'table_row':
                meta = node.get('node_meta', {})
                row_data = ' | '.join([f"{k}:{v}" for k, v in meta.items() if k != 'row_index'])
                label = f"Row {meta.get('row_index', '?')}\n{row_data[:30]}"
            else:
                label = f"{modality}\n{content[:30]}"
            ax.annotate(label, (x, y), ha='center', va='center',
                        fontsize=8, weight='bold', alpha=0.9)

        # Interactivity
        if interactive and MPLCURSORS_AVAILABLE:
            cursor = mplcursors.cursor(artists, hover=False)
            @cursor.connect("add")
            def on_add(sel):
                artist = sel.artist
                gid = artist.get_gid()
                if gid is None:
                    return
                if gid in self.node_map:
                    node = self.node_map[gid]
                    content = node.get('content', '')
                    if len(content) > 300:
                        content = content[:300] + '…'
                    text = (
                        f"ID: {gid[:8]}\n"
                        f"Modality: {node.get('modality', 'unknown')}\n"
                        f"Subgraph: {node.get('subgraph_id', 'N/A')}\n"
                        f"Content: {content}"
                    )
                elif gid in self.edge_map:
                    edge = self.edge_map[gid]
                    src = edge.get('source_id', '')[:8]
                    tgt = edge.get('target_id', '')[:8]
                    text = (
                        f"Edge ID: {gid[:8]}\n"
                        f"Type: {edge.get('type', 'unknown')}\n"
                        f"From: {src}\n"
                        f"To: {tgt}\n"
                        f"Meta: {edge.get('edge_meta', {})}"
                    )
                else:
                    text = f"Unknown element (gid={gid})"
                sel.annotation.set_text(text)
                sel.annotation.get_bbox_patch().set(fc="white", alpha=0.9)

        if output_path:
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
            print(f"✅ Subgraph saved to {output_path}")
        else:
            plt.show()

    # ---------- Interactive Explorer with Numbering & Save ----------
    def _ask_save(self, default_name: str = "figure.png"):
        """Ask user if they want to save the figure."""
        while True:
            resp = input(f"Save this figure? (y/n/custom_filename) [default: {default_name}]: ").strip()
            if resp.lower() in ('y', 'yes'):
                return default_name
            elif resp.lower() in ('n', 'no', ''):
                return None
            else:
                return resp  # custom filename

    def interactive_explorer(self, json_path: str):
        print("\n" + "="*60)
        print(f"📄 Loaded: {json_path}")
        print(f"   Nodes: {len(self.nodes)}, Edges: {len(self.edges)}")
        print("="*60)
        self.list_subgraphs()
        print("\n" + "-"*60)
        print("Commands:")
        print("  p [page_num]    – show page view (e.g. p 0)")
        print("  s [num]         – show subgraph by number (e.g. s 1)")
        print("  l               – list subgraphs again")
        print("  a               – save all pages as 'all_pages.png'")
        print("  q               – quit")
        print("-"*60)

        while True:
            cmd = input("\n> ").strip()
            if not cmd:
                continue
            parts = cmd.split()
            action = parts[0].lower()

            if action == 'q':
                print("👋 Goodbye!")
                break
            elif action == 'l':
                self.list_subgraphs()
            elif action == 'a':
                output = "all_pages.png"
                self.draw(mode='save_all', output_path=output, dpi=150)
                print(f"✅ All pages saved as {output}")
            elif action == 'p':
                page = int(parts[1]) if len(parts) > 1 else 0
                self.draw(page_num=page, mode='interactive')
                # Ask to save
                fname = self._ask_save(f"page_{page}.png")
                if fname:
                    self.draw(page_num=page, mode='save_all', output_path=fname)
                    print(f"✅ Saved as {fname}")
            elif action == 's':
                if len(parts) < 2:
                    print("Please provide subgraph number, e.g. s 1")
                    continue
                try:
                    num = int(parts[1])
                except ValueError:
                    print("Please enter a number.")
                    continue
                if num not in self.subgraph_index:
                    print(f"Subgraph number {num} not found. Use 'l' to list.")
                    continue
                sg_id = self.subgraph_index[num]
                self.draw_subgraph(sg_id)
            else:
                print("Unknown command. Use p, s, l, a, q.")

    def list_subgraphs(self):
        """Print numbered subgraphs with container info."""
        if not self.subgraph_index:
            print("No subgraphs found.")
            return
        print(f"\n📂 Found {len(self.subgraph_index)} subgraphs:")
        for num, sg_id in sorted(self.subgraph_index.items()):
            node_ids = self.subgraph_groups[sg_id]
            container = next(
                (self.node_map[nid] for nid in node_ids if self.node_map[nid].get('subgraph_role') == 'container'),
                None
            )
            modality = container.get('modality', 'unknown') if container else 'unknown'
            content = container.get('content', '')[:40] if container else ''
            page_info = f"Page {container['bbox']['page']+1}" if container and 'bbox' in container else "No page"
            print(f"  {num:3d}  {sg_id[:16]}  ({len(node_ids)} nodes, {modality})  {content}  {page_info}")


# ======== Main ========
if __name__ == "__main__":
    import sys
    json_file = sys.argv[1] if len(sys.argv) > 1 else "enriched_output_full.json"
    viz = DoclingGraphVisualizer(json_path=json_file)
    viz.interactive_explorer(json_file)