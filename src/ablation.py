"""
Ablation study - isolates the contribution of each detection component.

Research question: does the GNN itself provide useful detection
capability, or is observed performance mainly coming from Isolation
Forest and/or their combination?

────────────────────────────────────────────────────────────────────
Inspection finding, before any new code (task step 1) - this changes
what "Isolation Forest" means in this round
────────────────────────────────────────────────────────────────────
Read directly from evaluate.py and run_comparison.py, not assumed:

    gnn_score      = reconstruction_scores(model, cache, g, node_list)
                     mean(1 - sigmoid(Zq_i . Zk_j)) over each node's real
                     edges, where Z is the GNN's own trained embeddings
                     and Zq/Zk are its learned directional projections.

    if_score_on_Z  = isolation_scores(Z, seed)
                     IsolationForest(n_estimators=200, contamination=
                     "auto", random_state=seed).fit(Z); score =
                     -clf.score_samples(Z)
                     *** Z here is the GNN's OWN embeddings, not raw
                     features - this is the score used INSIDE the
                     existing "GNN + IF" combination. ***

    combined_score = combined_anomaly_score(gnn_score, if_score_on_Z)
                     = 0.5 * rank_normalize(gnn_score)
                     + 0.5 * rank_normalize(if_score_on_Z)
                     i.e. UNWEIGHTED RANK AVERAGING (not raw-score
                     averaging, not z-score-normalized averaging, not
                     weighted). rank_normalize(x) = argsort-based
                     percentile in [0, 1]. This exact formula is
                     unchanged this round, per instruction.

The consequence: the existing "GNN + IF" is not "GNN combined with an
independent classical method" - it's two scoring functions applied to
the SAME underlying GNN-learned representation. For Experiment B ("IF
only... do not use GNN reconstruction score") to be a genuinely
independent baseline - and for the leakage check "IF-only score is
exactly the same score that would exist without the GNN module" to
actually hold - Isolation Forest must be computed on the RAW,
pre-GNN feature matrix X (from features.py, unchanged), not on Z.

That is the one addition this module makes: `isolation_scores(X, seed)`
instead of `isolation_scores(Z, seed)` for Experiment B specifically.
Nothing about evaluate.py, gnn_model.py, or the existing "GNN + IF"
combination (Experiment C, computed exactly as before, still using
if_score_on_Z internally) is modified.

────────────────────────────────────────────────────────────────────
The four experiments
────────────────────────────────────────────────────────────────────
  A. GNN only              reconstruction_scores() - unchanged
  B. Isolation Forest only  isolation_scores(X, seed) - NEW: on raw
                            features, genuinely GNN-independent
  C. GNN + Isolation Forest  the EXISTING combination, unchanged -
                            internally still uses isolation_scores(Z,
                            seed), i.e. IF-on-GNN-embeddings, exactly
                            as before
  D. BankForensIQ Transaction Rules  combined_rule_score() - unchanged,
                            v5-corrected, reference baseline only
"""

import sys
import warnings
from pathlib import Path
from typing import Dict, List

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from graph_builder import build_multigraph, build_adjacency, build_directional_channels, directed_edge_index
from labeling import inject_synthetic_rings, TYPES, TIERS
from features import compute_node_features
from gnn_model import train_directional_gae
from evaluate import reconstruction_scores, isolation_scores, combined_anomaly_score
from rule_baseline import combined_rule_score
from metrics import compute_metrics

EXPERIMENTS = [
    "BankForensIQ Rules",
    "Isolation Forest",       # Experiment B: on RAW features (corrected/independent)
    "GNN",                    # Experiment A
    "GNN + IF",                # Experiment C: existing combination, unchanged
]


def _labels_for_slice(node_list, ring_records, slice_type=None, slice_tier=None):
    synthetic_in_slice = set()
    all_synthetic = set()
    for r in ring_records:
        all_synthetic.update(r["nodes"])
        if (slice_type is None or r["ring_type"] == slice_type) and \
           (slice_tier is None or r["tier"] == slice_tier):
            synthetic_in_slice.update(r["nodes"])
    labels = np.array([n in synthetic_in_slice for n in node_list])
    other_synthetic = np.array([(n in all_synthetic) and (n not in synthetic_in_slice) for n in node_list])
    return labels, ~other_synthetic


def run_one_seed_ablation(txns, seed, rings_per_type_tier=10, epochs=200,
                           hidden_dim=64, embed_dim=32, lr=0.02):
    """
    Builds the graph and trains the GNN exactly as run_comparison.py
    does (unchanged code, unchanged parameters), then scores all four
    experiments. Returns per-seed metrics for overall + per-type +
    per-tier slices, plus the raw score arrays' provenance for the
    leakage tests in tests/test_ablation.py.
    """
    g, _ = build_multigraph(txns)
    g, ring_records = inject_synthetic_rings(g, rings_per_type_tier=rings_per_type_tier, seed=seed)
    X, node_list, feat_df = compute_node_features(g)

    adj_sym, node_list2 = build_adjacency(g)
    assert node_list == node_list2
    channels = build_directional_channels(g, node_list)
    pos_i, pos_j = directed_edge_index(g, node_list)
    channels["_pos_i"], channels["_pos_j"] = pos_i, pos_j

    model, Z, cache, losses = train_directional_gae(
        X, channels, adj_sym, epochs=epochs, lr=lr,
        hidden_dim=hidden_dim, embed_dim=embed_dim, seed=seed, verbose=False,
    )

    # Experiment A - GNN only. Computed purely from the trained model's
    # own decoder; does not import or call sklearn/IsolationForest at all
    # (see reconstruction_scores in evaluate.py).
    gnn_score = reconstruction_scores(model, cache, g, node_list)

    # Experiment B - Isolation Forest only, on RAW features X (the
    # correction this round makes - see module docstring). This call
    # never touches Z, the trained model, or gnn_score.
    if_score_raw = isolation_scores(X, seed=seed)

    # Experiment C - the EXISTING "GNN + IF" combination, unchanged.
    # Internally recomputes IF on Z (the GNN's embeddings) - kept exactly
    # as the current implementation does it, not altered this round.
    if_score_on_Z = isolation_scores(Z, seed=seed)
    combined_score = combined_anomaly_score(gnn_score, if_score_on_Z)

    # Experiment D - reference baseline, unchanged from v5.
    rule_score = combined_rule_score(g, node_list)["combined"]

    scores = {
        "BankForensIQ Rules": rule_score,
        "Isolation Forest": if_score_raw,
        "GNN": gnn_score,
        "GNN + IF": combined_score,
    }

    labels_all, keep_all = _labels_for_slice(node_list, ring_records)
    overall = {m: compute_metrics(s[keep_all], labels_all[keep_all]) for m, s in scores.items()}

    by_type = {}
    for t in TYPES:
        labels_t, keep_t = _labels_for_slice(node_list, ring_records, slice_type=t)
        by_type[t] = {m: compute_metrics(s[keep_t], labels_t[keep_t]) for m, s in scores.items()}

    by_tier = {}
    for tier in TIERS:
        labels_te, keep_te = _labels_for_slice(node_list, ring_records, slice_tier=tier)
        by_tier[tier] = {m: compute_metrics(s[keep_te], labels_te[keep_te]) for m, s in scores.items()}

    return dict(
        seed=seed, n_nodes=g.number_of_nodes(), n_edges=g.number_of_edges(),
        n_rings=len(ring_records), final_loss=float(losses[-1]),
        overall=overall, by_type=by_type, by_tier=by_tier,
        # kept for the leakage/independence tests - NOT part of the report
        _node_list_len=len(node_list),
        _gnn_score_stats=dict(min=float(gnn_score.min()), max=float(gnn_score.max())),
        _if_raw_score_stats=dict(min=float(if_score_raw.min()), max=float(if_score_raw.max())),
    )


def _mean_std(vals):
    vals = [v for v in vals if v is not None]
    return (float(np.mean(vals)), float(np.std(vals))) if vals else (None, None)


def aggregate(per_seed_results, path_fn):
    agg = {}
    for method in EXPERIMENTS:
        aucs = [path_fn(r)[method]["auc"] for r in per_seed_results]
        prs = [path_fn(r)[method]["pr_auc"] for r in per_seed_results]
        precs = [path_fn(r)[method]["precision_at_k"] for r in per_seed_results]
        recs = [path_fn(r)[method]["recall_at_k"] for r in per_seed_results]
        auc_m, auc_s = _mean_std(aucs)
        pr_m, pr_s = _mean_std(prs)
        prec_m, prec_s = _mean_std(precs)
        rec_m, rec_s = _mean_std(recs)
        agg[method] = dict(auc_mean=auc_m, auc_std=auc_s, pr_auc_mean=pr_m, pr_auc_std=pr_s,
                            precision_at_k_mean=prec_m, precision_at_k_std=prec_s,
                            recall_at_k_mean=rec_m, recall_at_k_std=rec_s)
    return agg


def contribution_analysis(per_seed_results):
    """
    Paired per-seed deltas (not delta-of-means), aggregated as mean +/-
    std across the 5 seeds - the statistically correct way to compare
    two methods evaluated on the SAME seeds, and how item 12's "report
    mean +/- std, don't manufacture significance with n=5" is honored:
    no p-value is computed, just the paired deltas and a plain sign-
    consistency count (how many of the 5 seeds agreed on direction).
    """
    def deltas(method_a, method_b, metric):
        vals = []
        for r in per_seed_results:
            a = r["overall"][method_a][metric]
            b = r["overall"][method_b][metric]
            if a is not None and b is not None:
                vals.append(a - b)
        return vals

    def summarize(method_a, method_b):
        out = {}
        for metric in ("auc", "pr_auc", "precision_at_k", "recall_at_k"):
            d = deltas(method_a, method_b, metric)
            if not d:
                out[metric] = dict(mean=None, std=None, n_positive_seeds=None, n_seeds=0)
                continue
            mean, std = _mean_std(d)
            n_pos = sum(1 for x in d if x > 0)
            out[metric] = dict(mean=mean, std=std, n_positive_seeds=n_pos, n_seeds=len(d))
        return out

    return dict(
        gnn_vs_if=summarize("GNN", "Isolation Forest"),
        gnn_plus_if_vs_gnn=summarize("GNN + IF", "GNN"),
    )
