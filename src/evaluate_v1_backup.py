"""
Evaluation - Phase 5 from STATE.md.

Two complementary anomaly signals, combined by averaged rank:

  reconstruction_score  per node, mean (1 - predicted edge probability)
                         over its real edges - "how hard is this node's
                         actual connectivity to explain from structure
                         alone".
  isolation_score       sklearn IsolationForest over the embedding
                         space Z - "how much of an outlier is this
                         node's embedding vs the rest of the graph".

Recovery metrics then check whether the *injected synthetic* nodes rank
higher on the combined score than real nodes do - this is the only
ground truth available (see labeling.py), so it is a check of "does
this pipeline have any signal at all", not a validated fraud detector.
"""

from typing import Dict, List

import networkx as nx
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

from gnn_model import GCNAutoencoder, _sigmoid


def reconstruction_scores(model: GCNAutoencoder, Z: np.ndarray,
                           g: nx.MultiDiGraph, node_list: List[str]) -> np.ndarray:
    index = {n: i for i, n in enumerate(node_list)}
    n = len(node_list)
    surprise_sum = np.zeros(n, dtype=np.float64)
    surprise_count = np.zeros(n, dtype=np.float64)

    seen_pairs = set()
    for u, v in g.edges():
        i, j = index[u], index[v]
        key = (min(i, j), max(i, j))
        if key in seen_pairs:
            continue
        seen_pairs.add(key)

        p = _sigmoid(float(np.dot(Z[i], Z[j])))
        surprise = 1.0 - p
        surprise_sum[i] += surprise
        surprise_sum[j] += surprise
        surprise_count[i] += 1
        surprise_count[j] += 1

    surprise_count[surprise_count == 0] = 1
    return (surprise_sum / surprise_count).astype(np.float32)


def isolation_scores(Z: np.ndarray, seed: int = 0) -> np.ndarray:
    clf = IsolationForest(n_estimators=200, contamination="auto", random_state=seed)
    clf.fit(Z)
    raw = -clf.score_samples(Z)  # higher = more anomalous
    return raw.astype(np.float32)


def _rank_normalize(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x))
    return ranks / max(len(x) - 1, 1)


def combined_anomaly_score(recon: np.ndarray, iso: np.ndarray) -> np.ndarray:
    return 0.5 * _rank_normalize(recon) + 0.5 * _rank_normalize(iso)


def evaluate_recovery(anomaly_score: np.ndarray, node_list: List[str],
                       ring_records: List[dict]) -> Dict:
    is_synthetic = np.array([n.startswith("SYN:") for n in node_list])
    n_synthetic = int(is_synthetic.sum())

    auc = roc_auc_score(is_synthetic, anomaly_score) if 0 < n_synthetic < len(node_list) else None

    order = np.argsort(-anomaly_score)  # descending
    top_k = order[:n_synthetic]
    hits_at_k = int(is_synthetic[top_k].sum())
    precision_at_k = hits_at_k / n_synthetic if n_synthetic else None

    ranks = _rank_normalize(anomaly_score)
    mean_percentile_synthetic = float(ranks[is_synthetic].mean()) if n_synthetic else None
    mean_percentile_real = float(ranks[~is_synthetic].mean())

    per_ring = []
    index = {n: i for i, n in enumerate(node_list)}
    for record in ring_records:
        idxs = [index[n] for n in record["nodes"] if n in index]
        ring_scores = ranks[idxs]
        per_ring.append(dict(
            ring_id=record["ring_id"],
            ring_type=record["ring_type"],
            n_nodes=len(idxs),
            mean_percentile=float(ring_scores.mean()),
            max_percentile=float(ring_scores.max()),
        ))

    return dict(
        n_synthetic_nodes=n_synthetic,
        n_real_nodes=len(node_list) - n_synthetic,
        auc=auc,
        precision_at_k=precision_at_k,
        k=n_synthetic,
        mean_percentile_synthetic=mean_percentile_synthetic,
        mean_percentile_real=mean_percentile_real,
        per_ring=per_ring,
    )


def format_report(results: Dict) -> str:
    lines = [
        "Synthetic ring recovery (Label Strategy C validation)",
        "=" * 55,
        f"Synthetic nodes injected : {results['n_synthetic_nodes']}",
        f"Real nodes in graph      : {results['n_real_nodes']}",
        "",
        f"AUC (synthetic vs real)  : {results['auc']:.3f}" if results['auc'] is not None else "AUC: n/a",
        f"Precision@{results['k']:<3d}          : {results['precision_at_k']:.1%}"
        if results['precision_at_k'] is not None else "Precision@k: n/a",
        f"Mean percentile - synthetic nodes : {results['mean_percentile_synthetic']:.1%}"
        " (100% = most anomalous)",
        f"Mean percentile - real nodes      : {results['mean_percentile_real']:.1%}",
        "",
        "Per-ring breakdown (mean percentile rank of that ring's nodes):",
    ]
    for r in results["per_ring"]:
        lines.append(
            f"  ring {r['ring_id']:2d} [{r['ring_type']:8s}] "
            f"n={r['n_nodes']}  mean_percentile={r['mean_percentile']:.1%}  "
            f"max_percentile={r['max_percentile']:.1%}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    import pickle
    import sys
    sys.path.insert(0, ".")
    from gnn_model import GCNAutoencoder, normalize_adjacency
    from graph_builder import build_adjacency

    with open("_graph_cache.pkl", "rb") as f:
        g, node_list, ring_records, feat_df = pickle.load(f)

    Z = np.load("_Z_cache.npy")
    adj, node_list2 = build_adjacency(g)
    assert node_list == node_list2

    # rebuild a throwaway model just to reuse its forward/decoder math
    # (we only need Z, already computed, plus the sigmoid decoder)
    dummy = GCNAutoencoder(n_features=feat_df.shape[1], hidden_dim=1, embed_dim=Z.shape[1])

    recon = reconstruction_scores(dummy, Z, g, node_list)
    iso = isolation_scores(Z)
    combined = combined_anomaly_score(recon, iso)

    results = evaluate_recovery(combined, node_list, ring_records)
    print(format_report(results))

    np.save("_anomaly_score_cache.npy", combined)
