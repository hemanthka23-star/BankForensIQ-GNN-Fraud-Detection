"""
Multi-seed reproducibility benchmark - steps 2-5 from the second review.

    python run_benchmark.py --data-dir /path/to/Bank-statements-dataset

Ingestion (parsing all the real statement files) is deterministic and
the most expensive single step, so it happens ONCE; everything seed-
dependent (injection, feature computation on the now-different graph,
model init, negative sampling) is re-run fresh per seed by rebuilding
the graph from the cached transaction list - simpler and safer than
trying to reset/copy a mutated graph between runs.

Unlike run_pipeline.py (which writes full artifacts - graph.gpickle,
embeddings, subgraph images - for one detailed run), this script only
keeps the numeric results per seed, since 5x copies of those heavy
artifacts aren't useful. Writes:

    data/processed/benchmark_multiseed.json   full per-seed + aggregated results
    data/processed/benchmark_report.txt        human-readable summary
"""

import argparse
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
from evaluate import reconstruction_scores, isolation_scores, combined_anomaly_score, evaluate_recovery

DEFAULT_SEEDS = [42, 43, 44, 45, 46]


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

    recon = reconstruction_scores(model, cache, g, node_list)
    iso = isolation_scores(Z, seed=seed)
    combined = combined_anomaly_score(recon, iso)
    results = evaluate_recovery(combined, node_list, ring_records)

    results["final_loss"] = float(losses[-1])
    results["initial_loss"] = float(losses[0])
    results["n_nodes"] = g.number_of_nodes()
    results["n_edges"] = g.number_of_edges()
    results["seed"] = seed
    return results


def _mean_std(vals):
    vals = [v for v in vals if v is not None]
    return (float(np.mean(vals)), float(np.std(vals))) if vals else (None, None)


def aggregate(all_results):
    auc_mean, auc_std = _mean_std([r["auc"] for r in all_results])
    auc_core_mean, auc_core_std = _mean_std([r["auc_core_only"] for r in all_results])
    prec_mean, prec_std = _mean_std([r["precision_at_k"] for r in all_results])
    loss_mean, loss_std = _mean_std([r["final_loss"] for r in all_results])

    cells = defaultdict(list)
    for r in all_results:
        for row in r["benchmark_table"]:
            cells[(row["ring_type"], row["tier"])].append(row["mean_percentile_all_nodes"])

    table = []
    for (ring_type, tier), vals in sorted(cells.items()):
        m, s = _mean_std(vals)
        table.append(dict(ring_type=ring_type, tier=tier, mean_percentile=m,
                           std_percentile=s, n_seeds=len(vals)))

    return dict(
        n_seeds=len(all_results),
        seeds=[r["seed"] for r in all_results],
        auc_mean=auc_mean, auc_std=auc_std,
        auc_core_mean=auc_core_mean, auc_core_std=auc_core_std,
        precision_at_k_mean=prec_mean, precision_at_k_std=prec_std,
        final_loss_mean=loss_mean, final_loss_std=loss_std,
        benchmark_table=table,
    )


def format_pivot_table(agg: dict) -> str:
    """Fraud type (rows) x difficulty tier (columns), mean +/- std percentile."""
    tiers = ["easy", "medium", "hard"]
    cell = {(r["ring_type"], r["tier"]): r for r in agg["benchmark_table"]}

    lines = [f"{'Fraud Type':16s}" + "".join(f"{t.title():>18s}" for t in tiers)]
    for ring_type in TYPES:
        row = f"{ring_type:16s}"
        for tier in tiers:
            r = cell.get((ring_type, tier))
            if r:
                cell_text = f"{r['mean_percentile']:.1%} +/- {r['std_percentile']:.1%}"
            else:
                cell_text = "n/a"
            row += f"{cell_text:>18s}"
        lines.append(row)
    return "\n".join(lines)


def format_report(agg: dict) -> str:
    lines = [
        "Multi-seed reproducibility benchmark",
        "=" * 60,
        f"Seeds: {agg['seeds']}  (n={agg['n_seeds']})",
        "",
        f"AUC (all synthetic vs real)  : {agg['auc_mean']:.3f} +/- {agg['auc_std']:.3f}"
        if agg["auc_mean"] is not None else "AUC: n/a",
        f"AUC (core-only vs real)      : {agg['auc_core_mean']:.3f} +/- {agg['auc_core_std']:.3f}"
        if agg["auc_core_mean"] is not None else "AUC (core-only): n/a",
        f"Precision@k                  : {agg['precision_at_k_mean']:.1%} +/- {agg['precision_at_k_std']:.1%}"
        if agg["precision_at_k_mean"] is not None else "Precision@k: n/a",
        f"Final training loss          : {agg['final_loss_mean']:.3f} +/- {agg['final_loss_std']:.3f}",
        "",
        "Mean detectability by fraud type x difficulty tier (mean +/- std across seeds):",
        format_pivot_table(agg),
        "",
        "If 'hard' no longer scores above 'easy' here (compare to STATE.md's",
        "v2 finding), the tier/size confound fix worked.",
    ]
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

    print(f"Ingesting {args.data_dir} once (deterministic, reused across all seeds)...")
    t0 = time.time()
    report = ingest_directory(args.data_dir, verbose=False)
    print(f"{len(report.transactions)} transactions ({time.time() - t0:.1f}s)\n")

    all_results = []
    for seed in args.seeds:
        t0 = time.time()
        print(f"--- seed {seed} ---")
        results = run_one_seed(
            report.transactions, seed, args.rings_per_type_tier, args.epochs,
            args.hidden_dim, args.embed_dim, args.lr,
        )
        elapsed = time.time() - t0
        print(f"  nodes={results['n_nodes']} edges={results['n_edges']} "
              f"final_loss={results['final_loss']:.3f} "
              f"AUC={results['auc']:.3f} AUC_core={results['auc_core_only']:.3f} "
              f"({elapsed:.1f}s)")
        all_results.append(results)

    agg = aggregate(all_results)
    report_text = format_report(agg)
    print("\n" + report_text)

    with open(out_dir / "benchmark_report.txt", "w") as f:
        f.write(report_text + "\n")

    with open(out_dir / "benchmark_multiseed.json", "w") as f:
        json.dump(dict(
            config=dict(
                seeds=args.seeds, rings_per_type_tier=args.rings_per_type_tier,
                epochs=args.epochs, hidden_dim=args.hidden_dim,
                embed_dim=args.embed_dim, lr=args.lr,
                excel_adapter_used=EXCEL_ADAPTER_SOURCE,
                total_transactions=len(report.transactions),
            ),
            per_seed=all_results,
            aggregate=agg,
        ), f, indent=2, default=str)

    print(f"\nWrote {out_dir / 'benchmark_report.txt'} and benchmark_multiseed.json")


if __name__ == "__main__":
    main()
