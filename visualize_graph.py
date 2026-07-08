import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Circle
from collections import defaultdict
from typing import List, Dict, Any, Optional

try:
    import mplcursors
    MPLCURSORS_AVAILABLE = True
except ImportError:
    MPLCURSORS_AVAILABLE = False
    print("⚠️ mplcursors not installed. Install with: pip install mplcursors")

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
        "unknown": "white"
    }
    EDGE_COLORS = {
        "contains_page": "black",
        "contains_section": "brown",
        "contains": "grey",
        "captioned_by": "red",
        "read_order_next": "blue",
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
        else:
            self.nodes = nodes or []
            self.edges = edges or []
            self.doc = {}

        self.node_map = {n['id']: n for n in self.nodes}
        self.edge_map = {e['id']: e for e in self.edges if 'id' in e}

        self.page_groups = defaultdict(list)
        for n in self.nodes:
            if 'bbox' in n and n['bbox'] is not None:
                self.page_groups[n['bbox']['page']].append(n)

        self.page_dims = {}
        for n in self.nodes:
            if n.get('modality') == 'page' and n.get('node_meta'):
                p = n['bbox']['page']
                self.page_dims[p] = (n['node_meta']['width'], n['node_meta']['height'])

        # Cross‑page connections for badges
        self.node_cross_connections = defaultdict(set)
        for edge in self.edges:
            src = self.node_map.get(edge['source_id'])
            tgt = self.node_map.get(edge['target_id'])
            if not src or not tgt:
                continue
            if src['bbox']['page'] != tgt['bbox']['page']:
                self.node_cross_connections[edge['source_id']].add(tgt['bbox']['page'])
                self.node_cross_connections[edge['target_id']].add(src['bbox']['page'])

    def _get_node_color(self, node):
        return self.NODE_COLORS.get(node.get('modality', 'unknown'), 'white')

    def _get_edge_color(self, edge):
        return self.EDGE_COLORS.get(edge.get('type', 'default'), self.EDGE_COLORS['default'])

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

        sx = (src['bbox']['x'] + src['bbox']['w']/2) * page_width
        sy = (src['bbox']['y'] + src['bbox']['h']/2) * page_height
        tx = (tgt['bbox']['x'] + tgt['bbox']['w']/2) * page_width
        ty = (tgt['bbox']['y'] + tgt['bbox']['h']/2) * page_height

        color = self._get_edge_color(edge)
        edge_id = edge.get('id')
        line_kw = dict(color=color, linewidth=2.5, alpha=0.8, gid=edge_id)

        if draw_arrow and edge.get('type') == 'read_order_next':
            # Draw arrow with picker support
            arrow = FancyArrowPatch((sx, sy), (tx, ty),
                                    arrowstyle='->', mutation_scale=15,
                                    **line_kw,
                                    picker=True)  # enable picking
            ax.add_patch(arrow)
            arrow.set_gid(edge_id)
            arrow.set_picker(5)  # set pick radius

            # Transparent overlay for better click detection
            overlay, = ax.plot([sx, tx], [sy, ty],
                               color=color, linewidth=8, alpha=0.0,
                               gid=edge_id, picker=True)
        else:
            # Regular line with picker
            line, = ax.plot([sx, tx], [sy, ty], linestyle='-', picker=True, **line_kw)
            line.set_gid(edge_id)

        # Clickable circle at midpoint
        mx = (sx + tx) / 2
        my = (sy + ty) / 2
        circle = Circle((mx, my), radius=5,
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

            # Collect all pickable artists with gid from this axes
            for artist in ax.get_children():
                if hasattr(artist, 'get_gid') and artist.get_gid() is not None:
                    # Check if it has picker support (either picker attr or isinstance check)
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


if __name__ == "__main__":
    viz = DoclingGraphVisualizer(json_path="parser_output_full.json")
    viz.draw(page_num=1, mode='interactive', show_read_order_arrows=True,
             show_cross_badge=True, show_sequence_numbers=False)
    # Batch: save all pages as a static PNG (no interactivity)
    #viz.draw(mode='save_all', output_path="all_pages_static.png", dpi=150)