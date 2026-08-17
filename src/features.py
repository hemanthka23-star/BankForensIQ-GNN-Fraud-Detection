"""
Feature engineering over the transaction graph.

Produces a node feature matrix X (numpy, aligned to a node_list) plus a
human-readable per-node feature table (pandas), computed from the
MultiDiGraph built by graph_builder.build_multigraph.

All features are derived purely from transaction structure/timing/amount
- nothing here looks at names, UPI handles, or any other PII, so the
resulting feature matrix itself carries no personally identifying
information even though the graph nodes it's aligned to do.
"""

from collections import defaultdict
from datetime import datetime
from typing import List, Tuple

import networkx as nx
import numpy as np
import pandas as pd

FEATURE_NAMES = [
    "is_account",
    "degree",
    "in_degree",
    "out_degree",
    "txn_count",
    "total_amount_in",
    "total_amount_out",
    "net_flow",
    "mean_amount",
    "std_amount",
    "max_amount",
    "unique_counterparties",
    "unique_banks",
    "unique_payment_methods",
    "reversal_count",
    "active_days",
    "date_span_days",
    "txns_per_active_day",
    "weekend_ratio",
]


def _safe_std(values):
    return float(np.std(values)) if len(values) > 1 else 0.0


def compute_node_features(g: nx.MultiDiGraph) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    node_list = list(g.nodes())
    rows = []

    for node in node_list:
        attrs = g.nodes[node]
        is_account = 1.0 if attrs.get("node_type") == "account" else 0.0

        in_edges = list(g.in_edges(node, data=True))
        out_edges = list(g.out_edges(node, data=True))

        in_degree = len(in_edges)
        out_degree = len(out_edges)
        txn_count = in_degree + out_degree

        amounts_in = [d["amount"] for _, _, d in in_edges]
        amounts_out = [d["amount"] for _, _, d in out_edges]
        all_amounts = amounts_in + amounts_out

        total_in = float(sum(amounts_in))
        total_out = float(sum(amounts_out))

        neighbors = set(u for u, _, _ in in_edges) | set(v for _, v, _ in out_edges)
        banks = set(
            g.nodes[u].get("bank") for u, _, _ in in_edges if g.nodes[u].get("bank")
        ) | set(
            g.nodes[v].get("bank") for _, v, _ in out_edges if g.nodes[v].get("bank")
        )
        methods = set(d.get("payment_method") for _, _, d in in_edges + out_edges)
        methods.discard(None)

        reversal_count = sum(
            1 for _, _, d in in_edges + out_edges if d.get("is_reversal")
        )

        dates = [
            d["parsed_date"] for _, _, d in in_edges + out_edges
            if d.get("parsed_date") is not None
        ]
        active_days = len(set(d.date() for d in dates))
        date_span_days = (max(dates) - min(dates)).days if len(dates) > 1 else 0
        weekend_count = sum(1 for d in dates if d.weekday() >= 5)
        weekend_ratio = weekend_count / len(dates) if dates else 0.0

        rows.append(dict(
            node_id=node,
            node_type=attrs.get("node_type"),
            label=attrs.get("label"),
            is_synthetic=attrs.get("is_synthetic", False),
            is_account=is_account,
            degree=txn_count,
            in_degree=in_degree,
            out_degree=out_degree,
            txn_count=txn_count,
            total_amount_in=total_in,
            total_amount_out=total_out,
            net_flow=total_in - total_out,
            mean_amount=float(np.mean(all_amounts)) if all_amounts else 0.0,
            std_amount=_safe_std(all_amounts),
            max_amount=float(max(all_amounts)) if all_amounts else 0.0,
            unique_counterparties=len(neighbors),
            unique_banks=len(banks),
            unique_payment_methods=len(methods),
            reversal_count=reversal_count,
            active_days=active_days,
            date_span_days=date_span_days,
            txns_per_active_day=txn_count / active_days if active_days else float(txn_count),
            weekend_ratio=weekend_ratio,
        ))

    df = pd.DataFrame(rows)

    X = df[FEATURE_NAMES].to_numpy(dtype=np.float32)

    # log1p-scale every heavy-tailed column, then z-score everything.
    # Real transaction graphs have hub accounts with vastly higher degree
    # than everything else (one account in this dataset alone has 10,875
    # transactions) - z-scoring raw counts under that kind of skew put
    # single nodes 60+ std out on several columns, which was destabilizing
    # GNN training (large activations -> saturated sigmoid -> loss spikes).
    # Originally only the amount columns were log1p'd; count-based columns
    # need the same treatment and didn't get it - fixed here.
    heavy_tail_cols = [
        FEATURE_NAMES.index(c) for c in
        ("degree", "in_degree", "out_degree", "txn_count",
         "total_amount_in", "total_amount_out", "net_flow", "mean_amount",
         "std_amount", "max_amount", "unique_counterparties", "unique_banks",
         "unique_payment_methods", "reversal_count", "active_days",
         "date_span_days", "txns_per_active_day")
    ]
    for c in heavy_tail_cols:
        X[:, c] = np.sign(X[:, c]) * np.log1p(np.abs(X[:, c]))

    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    X_norm = (X - mean) / std

    # Safety net: even after log1p, a near-constant column (almost every
    # node has reversal_count=0, say) has tiny std, so the rare nonzero
    # node can still land 50+ std out. Clip rather than let any single
    # column silently dominate the GNN's input magnitude.
    X_norm = np.clip(X_norm, -8.0, 8.0)

    return X_norm.astype(np.float32), node_list, df


if __name__ == "__main__":
    import pickle
    import sys
    sys.path.insert(0, ".")
    from graph_builder import build_multigraph

    with open("_txns_cache.pkl", "rb") as f:
        txns = pickle.load(f)

    g, report = build_multigraph(txns)
    X, node_list, df = compute_node_features(g)

    print(f"Feature matrix: {X.shape}")
    print(df.drop(columns=["label"]).describe().T[["mean", "std", "min", "max"]])
