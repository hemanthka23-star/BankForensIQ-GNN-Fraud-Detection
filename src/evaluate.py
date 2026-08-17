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

from gnn_model import DirectionalGCNAutoencoder


def reconstruction_scores(model: DirectionalGCNAutoencoder, cache: dict,
                           g: nx.MultiDiGraph, node_list: List[str]) -> np.ndarray:
    """
    Per node, mean (1 - predicted probability) over its real DIRECTED
    edges, using the model's actual (direction-aware) decoder rather
    than a raw symmetric dot product - an edge i->j and its reverse
    j->i can now legitimately get different reconstruction scores.
    """
    index = {n: i for i, n in enumerate(node_list)}
    n = len(node_list)
    surprise_sum = np.zeros(n, dtype=np.float64)
    surprise_count = np.zeros(n, dtype=np.float64)

    edge_list = list(g.edges())
    if not edge_list:
        return np.zeros(n, dtype=np.float32)

    i_idx = np.array([index[u] for u, v in edge_list])
    j_idx = np.array([index[v] for u, v in edge_list])
    p = model.decode_pairs(cache, i_idx, j_idx)
    surprise = 1.0 - p

    np.add.at(surprise_sum, i_idx, surprise)
    np.add.at(surprise_count, i_idx, 1.0)
    np.add.at(surprise_sum, j_idx, surprise)
    np.add.at(surprise_count, j_idx, 1.0)

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

    index = {n: i for i, n in enumerate(node_list)}

    per_ring = []
    for record in ring_records:
        idxs = [index[n] for n in record["nodes"] if n in index]
        core_idxs = [index[n] for n in record.get("core_nodes", record["nodes"]) if n in index]
        ring_scores = ranks[idxs]
        core_scores = ranks[core_idxs] if core_idxs else ring_scores
        per_ring.append(dict(
            ring_id=record["ring_id"],
            ring_type=record["ring_type"],
            tier=record.get("tier", "n/a"),
            n_nodes=len(idxs),
            n_core=len(core_idxs),
            mean_percentile=float(ring_scores.mean()),
            max_percentile=float(ring_scores.max()),
            mean_percentile_core=float(core_scores.mean()),
        ))

    # slice by (type, tier) - the systematic-benchmark table
    by_type_tier = {}
    for r in per_ring:
        key = (r["ring_type"], r["tier"])
        by_type_tier.setdefault(key, []).append(r)

    benchmark_table = []
    for (ring_type, tier), rows in sorted(by_type_tier.items()):
        benchmark_table.append(dict(
            ring_type=ring_type,
            tier=tier,
            n_rings=len(rows),
            mean_percentile_all_nodes=float(np.mean([r["mean_percentile"] for r in rows])),
            mean_percentile_core_nodes=float(np.mean([r["mean_percentile_core"] for r in rows])),
        ))

    # core-only AUC/precision - are the *specifically* anomalous nodes
    # (not just any ring member, including blended-in noise nodes)
    # findable? Stricter and more honest than the all-nodes number.
    core_node_set = set()
    for r in ring_records:
        core_node_set.update(r.get("core_nodes", r["nodes"]))
    is_core = np.array([n in core_node_set for n in node_list])
    non_core_synthetic = is_synthetic & ~is_core
    # for a fair core-only AUC, exclude non-core synthetic nodes entirely
    # (they're deliberately meant to sometimes look normal)
    keep_mask = ~non_core_synthetic
    auc_core_only = (
        roc_auc_score(is_core[keep_mask], anomaly_score[keep_mask])
        if 0 < is_core.sum() < keep_mask.sum() else None
    )

    return dict(
        n_synthetic_nodes=n_synthetic,
        n_real_nodes=len(node_list) - n_synthetic,
        n_core_nodes=int(is_core.sum()),
        auc=auc,
        auc_core_only=auc_core_only,
        precision_at_k=precision_at_k,
        k=n_synthetic,
        mean_percentile_synthetic=mean_percentile_synthetic,
        mean_percentile_real=mean_percentile_real,
        per_ring=per_ring,
        benchmark_table=benchmark_table,
    )


def format_report(results: Dict) -> str:
    lines = [
        "Synthetic ring recovery (Label Strategy C validation)",
        "=" * 55,
        f"Synthetic nodes injected : {results['n_synthetic_nodes']}  "
        f"(of which core/definitely-anomalous: {results['n_core_nodes']})",
        f"Real nodes in graph      : {results['n_real_nodes']}",
        "",
        f"AUC (all synthetic vs real)      : {results['auc']:.3f}" if results['auc'] is not None else "AUC: n/a",
        f"AUC (core-only vs real)          : {results['auc_core_only']:.3f}"
        " <- stricter: excludes ring nodes deliberately meant to blend in"
        if results['auc_core_only'] is not None else "AUC (core-only): n/a",
        f"Precision@{results['k']:<3d}                  : {results['precision_at_k']:.1%}"
        if results['precision_at_k'] is not None else "Precision@k: n/a",
        f"Mean percentile - synthetic nodes : {results['mean_percentile_synthetic']:.1%}"
        " (100% = most anomalous)",
        f"Mean percentile - real nodes      : {results['mean_percentile_real']:.1%}",
        "",
        "Benchmark table (mean percentile rank, by fraud type x difficulty tier):",
        f"  {'type':16s} {'tier':7s} {'n':>3s}  {'all-nodes':>10s}  {'core-only':>10s}",
    ]
    for row in results.get("benchmark_table", []):
        lines.append(
            f"  {row['ring_type']:16s} {row['tier']:7s} {row['n_rings']:3d}  "
            f"{row['mean_percentile_all_nodes']:9.1%}  {row['mean_percentile_core_nodes']:9.1%}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    print("evaluate.py imports OK. Run `python run_pipeline.py --data-dir ...` "
          "for a real end-to-end run (this module is exercised there).")
