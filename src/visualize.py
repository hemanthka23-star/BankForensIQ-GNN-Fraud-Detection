"""
Render a handful of subgraphs for visual sanity-checking (Phase 3 / 5:
"synthetic rings visibly recoverable by eye"). Real node labels (names,
UPI handles, etc.) are replaced with anonymized IDs (ACC_1, CP_42, ...)
before rendering - the point of the picture is the *structure* the
model is scoring, not the underlying PII. Node color encodes the
combined anomaly score; synthetic ring nodes are drawn as squares.
"""

import hashlib
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def _anon_label(node_id: str) -> str:
    if node_id.startswith("SYN:"):
        return node_id.replace("SYN:", "")
    prefix = "ACC" if node_id.startswith("ACC:") else "CP"
    h = hashlib.sha256(node_id.encode()).hexdigest()[:4].upper()
    return f"{prefix}_{h}"


def render_ring_subgraph(g: nx.MultiDiGraph, node_list: List[str],
                          anomaly_score: np.ndarray, ring_record: dict,
                          out_path: str, hops: int = 1):
    index = {n: i for i, n in enumerate(node_list)}
    ring_nodes = set(ring_record["nodes"])

    context = set(ring_nodes)
    undirected = g.to_undirected()
    for node in list(ring_nodes):
        context.update(nx.single_source_shortest_path_length(undirected, node, cutoff=hops).keys())

    sub = g.subgraph(context).copy()

    pos = nx.spring_layout(sub, seed=7, k=0.9)

    fig, ax = plt.subplots(figsize=(8, 6))

    node_colors = []
    node_shapes_square, node_shapes_circle = [], []
    for n in sub.nodes():
        score = anomaly_score[index[n]] if n in index else 0.5
        node_colors.append(score)
        (node_shapes_square if n in ring_nodes else node_shapes_circle).append(n)

    cmap = plt.cm.YlOrRd
    norm = plt.Normalize(vmin=0, vmax=1)

    circ_colors = [anomaly_score[index[n]] for n in node_shapes_circle]
    sq_colors = [anomaly_score[index[n]] for n in node_shapes_square]

    nx.draw_networkx_nodes(sub, pos, nodelist=node_shapes_circle, node_shape="o",
                            node_color=circ_colors, cmap=cmap, vmin=0, vmax=1,
                            node_size=500, ax=ax, edgecolors="black", linewidths=0.6)
    nx.draw_networkx_nodes(sub, pos, nodelist=node_shapes_square, node_shape="s",
                            node_color=sq_colors, cmap=cmap, vmin=0, vmax=1,
                            node_size=650, ax=ax, edgecolors="black", linewidths=1.4)

    edge_colors = ["crimson" if d.get("is_synthetic") else "lightgray"
                   for _, _, d in sub.edges(data=True)]
    nx.draw_networkx_edges(sub, pos, edge_color=edge_colors, arrows=True,
                            arrowsize=12, ax=ax, connectionstyle="arc3,rad=0.08")

    labels = {n: _anon_label(n) for n in sub.nodes()}
    nx.draw_networkx_labels(sub, pos, labels=labels, font_size=8, ax=ax)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.04)
    cbar.set_label("anomaly score (percentile rank)")

    ax.set_title(
        f"Ring {ring_record['ring_id']} [{ring_record['ring_type']}] "
        f"- squares = injected synthetic nodes, red edges = injected transactions",
        fontsize=10,
    )
    ax.axis("off")
    fig.subplots_adjust(left=0.03, right=0.97, top=0.90, bottom=0.03)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    import pickle
    import sys

    with open("_graph_cache.pkl", "rb") as f:
        g, node_list, ring_records, feat_df = pickle.load(f)

    anomaly_score = np.load("_anomaly_score_cache.npy")

    out_dir = Path("data/processed/subgraphs")
    out_dir.mkdir(parents=True, exist_ok=True)

    chosen = {}
    for r in ring_records:
        chosen.setdefault(r["ring_type"], r)

    for ring_type, record in chosen.items():
        out_path = out_dir / f"ring_{record['ring_id']}_{ring_type}.png"
        render_ring_subgraph(g, node_list, anomaly_score, record, str(out_path))
        print(f"wrote {out_path}")
