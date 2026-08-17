"""
Existing BankForensIQ graph rules vs Isolation Forest vs GNN vs GNN+IF,
on the exact same 150-ring, 5-seed benchmark used throughout this
project.

    python run_comparison.py --data-dir /path/to/Bank-statements-dataset

All four methods are scored on the SAME graph for a given seed (built
once, not rebuilt per method), using the SAME compute_metrics()
implementation (src/metrics.py), so the comparison is apples-to-apples.
The rule baseline never sees the GNN's score or the synthetic labels
during scoring - see tests/test_rule_baseline.py for the checks that
enforce this.

Writes to data/processed/:
    baseline_results.json     full per-seed, per-method, per-slice results
    comparison_report.txt     the three human-readable tables + the
                               required "not validated real-fraud
                               detection" disclaimer
    comparison_table.csv      the overall (method x metric) table
"""

import argparse
import csv
import json
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np

from ingest import ingest_directory, EXCEL_ADAPTER_SOURCE
from graph_builder import build_multigraph, build_adjacency, build_directional_channels, directed_edge_index
from labeling import inject_synthetic_rings, TYPES, TIERS
from features import compute_node_features
from gnn_model import train_directional_gae
from evaluate import reconstruction_scores, isolation_scores, combined_anomaly_score
from rule_baseline import combined_rule_score
from metrics import compute_metrics, format_metrics

DEFAULT_SEEDS = [42, 43, 44, 45, 46]
METHODS = ["BankForensIQ Transaction Rules", "Isolation Forest", "Graph-Structural GNN", "Graph-Structural GNN + Isolation Forest"]


def _labels_for_slice(node_list, ring_records, slice_type=None, slice_tier=None):
    """
    Boolean label array + a keep-mask (real nodes, plus only the
    synthetic nodes belonging to this slice - other-type/tier synthetic
    nodes are excluded from the comparison entirely for that slice, so
    they can't count as false positives/negatives for a slice they
    don't belong to).
    """
    synthetic_nodes_in_slice = set()
    all_synthetic_nodes = set()
    for r in ring_records:
        all_synthetic_nodes.update(r["nodes"])
        if (slice_type is None or r["ring_type"] == slice_type) and \
           (slice_tier is None or r["tier"] == slice_tier):
            synthetic_nodes_in_slice.update(r["nodes"])

    labels = np.array([n in synthetic_nodes_in_slice for n in node_list])
    other_synthetic = np.array([
        (n in all_synthetic_nodes) and (n not in synthetic_nodes_in_slice) for n in node_list
    ])
    keep_mask = ~other_synthetic
    return labels, keep_mask


def run_one_seed(txns, seed, rings_per_type_tier, epochs, hidden_dim, embed_dim, lr):
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

    gnn_score = reconstruction_scores(model, cache, g, node_list)
    if_score = isolation_scores(Z, seed=seed)
    combined_score = combined_anomaly_score(gnn_score, if_score)
    rule_scores = combined_rule_score(g, node_list)
    rule_score = rule_scores["combined"]

    scores_by_method = {
        "BankForensIQ Transaction Rules": rule_score,
        "Isolation Forest": if_score,
        "Graph-Structural GNN": gnn_score,
        "Graph-Structural GNN + Isolation Forest": combined_score,
    }

    # Diagnostic per task step 6: confirms whether an account-only
    # comparison would even be computable under this benchmark design.
    account_nodes = [n for n, d in g.nodes(data=True) if d.get("node_type") == "account"]
    synthetic_account_nodes = [n for n in account_nodes if n.startswith("SYN:")]

    # overall slice
    labels_all, keep_all = _labels_for_slice(node_list, ring_records)
    overall = {
        m: compute_metrics(s[keep_all], labels_all[keep_all])
        for m, s in scores_by_method.items()
    }

    by_type = {}
    for t in TYPES:
        labels_t, keep_t = _labels_for_slice(node_list, ring_records, slice_type=t)
        by_type[t] = {
            m: compute_metrics(s[keep_t], labels_t[keep_t])
            for m, s in scores_by_method.items()
        }

    by_tier = {}
    for tier in TIERS:
        labels_te, keep_te = _labels_for_slice(node_list, ring_records, slice_tier=tier)
        by_tier[tier] = {
            m: compute_metrics(s[keep_te], labels_te[keep_te])
            for m, s in scores_by_method.items()
        }

    return dict(
        seed=seed, n_nodes=g.number_of_nodes(), n_edges=g.number_of_edges(),
        n_rings=len(ring_records), final_loss=float(losses[-1]),
        n_account_nodes=len(account_nodes), n_synthetic_account_nodes=len(synthetic_account_nodes),
        overall=overall, by_type=by_type, by_tier=by_tier,
    )


def _mean_std(vals):
    vals = [v for v in vals if v is not None]
    return (float(np.mean(vals)), float(np.std(vals))) if vals else (None, None)


def aggregate_metric_cells(per_seed_results, path_fn):
    """path_fn(seed_result) -> {method: metrics_dict}; aggregates AUC/PR-AUC/Precision@K/Recall@K
    across seeds for each method."""
    agg = {}
    for method in METHODS:
        aucs = [path_fn(r)[method]["auc"] for r in per_seed_results]
        prs = [path_fn(r)[method]["pr_auc"] for r in per_seed_results]
        precs = [path_fn(r)[method]["precision_at_k"] for r in per_seed_results]
        recs = [path_fn(r)[method]["recall_at_k"] for r in per_seed_results]
        auc_m, auc_s = _mean_std(aucs)
        pr_m, pr_s = _mean_std(prs)
        prec_m, prec_s = _mean_std(precs)
        rec_m, rec_s = _mean_std(recs)
        agg[method] = dict(
            auc_mean=auc_m, auc_std=auc_s, pr_auc_mean=pr_m, pr_auc_std=pr_s,
            precision_at_k_mean=prec_m, precision_at_k_std=prec_s,
            recall_at_k_mean=rec_m, recall_at_k_std=rec_s,
        )
    return agg


def format_overall_table(agg_overall):
    lines = [f"{'Method':24s}{'AUC':>16s}{'PR-AUC':>16s}{'Precision@K':>16s}{'Recall@K':>16s}"]
    for method in METHODS:
        a = agg_overall[method]
        def fmt(m, s, pct=False):
            if m is None:
                return "n/a"
            return f"{m:.1%} +/- {s:.1%}" if pct else f"{m:.3f} +/- {s:.3f}"
        lines.append(
            f"{method:24s}{fmt(a['auc_mean'], a['auc_std']):>16s}"
            f"{fmt(a['pr_auc_mean'], a['pr_auc_std']):>16s}"
            f"{fmt(a['precision_at_k_mean'], a['precision_at_k_std'], True):>16s}"
            f"{fmt(a['recall_at_k_mean'], a['recall_at_k_std'], True):>16s}"
        )
    return "\n".join(lines)


def format_slice_table(title, keys, agg_by_key, metric="pr_auc_mean", metric_std="pr_auc_std"):
    lines = [f"\n{title} (metric: PR-AUC, mean +/- std across seeds)",
             f"{'':16s}" + "".join(f"{m[:14]:>17s}" for m in METHODS)]
    for key in keys:
        row = f"{str(key):16s}"
        for method in METHODS:
            a = agg_by_key[key][method]
            m, s = a.get(metric), a.get(metric_std)
            cell = f"{m:.1%}+/-{s:.1%}" if m is not None else "n/a"
            row += f"{cell:>17s}"
        lines.append(row)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--rings-per-type-tier", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.02)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Ingesting {args.data_dir} once...")
    t0 = time.time()
    report = ingest_directory(args.data_dir, verbose=False)
    print(f"{len(report.transactions)} transactions ({time.time() - t0:.1f}s)\n")

    per_seed_results = []
    for seed in args.seeds:
        t0 = time.time()
        r = run_one_seed(report.transactions, seed, args.rings_per_type_tier,
                          args.epochs, args.hidden_dim, args.embed_dim, args.lr)
        elapsed = time.time() - t0
        print(f"seed {seed}: nodes={r['n_nodes']} rings={r['n_rings']} "
              f"loss={r['final_loss']:.3f} "
              f"GNN_AUC={r['overall']['Graph-Structural GNN']['auc']:.3f} "
              f"Rules_AUC={r['overall']['BankForensIQ Transaction Rules']['auc']:.3f} "
              f"({elapsed:.1f}s)")
        per_seed_results.append(r)

    agg_overall = aggregate_metric_cells(per_seed_results, lambda r: r["overall"])
    agg_by_type = {t: aggregate_metric_cells(per_seed_results, lambda r, t=t: r["by_type"][t]) for t in TYPES}
    agg_by_tier = {tier: aggregate_metric_cells(per_seed_results, lambda r, tier=tier: r["by_tier"][tier]) for tier in TIERS}

    overall_table = format_overall_table(agg_overall)
    type_table = format_slice_table("Per-fraud-type comparison", TYPES, agg_by_type)
    tier_table = format_slice_table("Per-tier comparison", list(TIERS.keys()), agg_by_tier)

    disclaimer = (
        "\nThese results evaluate recovery of synthetically injected fraud structures.\n"
        "They are not measurements of confirmed real-world fraud detection because the\n"
        "real dataset has no fraud ground-truth labels."
    )

    account_note = (
        f"\nAccount-node limitation (task step 6): this dataset has "
        f"{per_seed_results[0]['n_account_nodes']} real account nodes; the fraud generator "
        f"never labels an account-type node as synthetic (every injected ring node is "
        f"node_type='counterparty' by construction), confirmed empirically: "
        f"{per_seed_results[0]['n_synthetic_account_nodes']} synthetic account nodes across "
        f"all seeds. An account-only comparison would have zero positive labels and no "
        f"computable AUC/PR-AUC under this benchmark design - not run, reported as a "
        f"limitation rather than worked around."
    )

    rule_mapping_note = (
        "\nBankForensIQ Transaction Rules mapping (task step 4-5): of the 7 real rules in "
        "backend/services/risk_engine.py, 5 are faithfully mapped to node-level scores "
        "(burst/RAPID_TRANSACTION, high_value/HIGH_VALUE_TRANSACTION, "
        "spending_spike/SPENDING_SPIKE, repeated_transaction/REPEATED_TRANSACTION, plus "
        "smurfing_episode from unified_fraud_engine.py's separate 'Smurfing' label). 3 are "
        "structurally unmappable to this graph (excessive_withdrawal needs narration text "
        "and ATM/cash transactions have no counterparty, so they're excluded from the graph "
        "entirely; late_night needs time-of-day data the graph doesn't carry; balance_drop "
        "needs per-transaction balance the graph doesn't carry) and score 0.0 for every node "
        "rather than being approximated. circular_return and bidirectional are confirmed "
        "absent from the codebase entirely, not just unmapped - also 0.0 for every node. "
        "See src/rule_baseline.py's module docstring for the full determination."
    )

    report_text = (
        "BankForensIQ Transaction Rules vs Isolation Forest vs Graph-Structural GNN vs GNN+IF\n" + "=" * 70 + "\n"
        f"Seeds: {args.seeds}  Rings: {args.rings_per_type_tier * len(TYPES) * len(TIERS)}  "
        f"Nodes (seed {args.seeds[0]}): {per_seed_results[0]['n_nodes']}\n\n"
        f"{overall_table}\n{type_table}\n{tier_table}\n{disclaimer}\n{account_note}\n{rule_mapping_note}\n"
    )
    print("\n" + report_text)

    with open(out_dir / "comparison_report.txt", "w") as f:
        f.write(report_text)

    with open(out_dir / "comparison_table.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "auc_mean", "auc_std", "pr_auc_mean", "pr_auc_std",
                          "precision_at_k_mean", "precision_at_k_std", "recall_at_k_mean", "recall_at_k_std"])
        for method in METHODS:
            a = agg_overall[method]
            writer.writerow([method, a["auc_mean"], a["auc_std"], a["pr_auc_mean"], a["pr_auc_std"],
                              a["precision_at_k_mean"], a["precision_at_k_std"],
                              a["recall_at_k_mean"], a["recall_at_k_std"]])

    with open(out_dir / "baseline_results.json", "w") as f:
        json.dump(dict(
            config=dict(
                seeds=args.seeds, rings_per_type_tier=args.rings_per_type_tier,
                epochs=args.epochs, hidden_dim=args.hidden_dim, embed_dim=args.embed_dim,
                lr=args.lr, excel_adapter_used=EXCEL_ADAPTER_SOURCE,
                total_transactions=len(report.transactions),
                rule_thresholds=dict(
                    rapid_window_minutes=10, rapid_burst_threshold=4, rapid_amount_floor=10000.0,
                    smurf_amount_low=9000.0, smurf_amount_high=9999.0,
                ),
                metric_definitions="src/metrics.py:compute_metrics",
            ),
            per_seed=per_seed_results,
            aggregate_overall=agg_overall,
            aggregate_by_type=agg_by_type,
            aggregate_by_tier=agg_by_tier,
        ), f, indent=2, default=str)

    print(f"\nWrote comparison_report.txt, comparison_table.csv, baseline_results.json to {out_dir}")


if __name__ == "__main__":
    main()
