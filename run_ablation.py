"""
Ablation study: isolates the contribution of GNN, Isolation Forest (on
raw features, genuinely independent - see src/ablation.py docstring for
why this differs from the "IF" inside the existing GNN+IF combination),
their existing combination, and the BankForensIQ rules reference, on
the SAME unchanged benchmark (150 rings, 5 types, 3 tiers, seeds
42-46).

    python run_ablation.py --data-dir /path/to/Bank-statements-dataset

Writes to data/processed/:
    ablation_results.json      full per-seed, per-method, per-slice results
    ablation_report.txt         sections A-H, per the task spec
    ablation_pr_auc_by_tier.png the one required figure
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ingest import ingest_directory, EXCEL_ADAPTER_SOURCE
from labeling import TYPES, TIERS
from ablation import run_one_seed_ablation, aggregate, contribution_analysis, EXPERIMENTS

DEFAULT_SEEDS = [42, 43, 44, 45, 46]


def format_table(title, agg, metric_mean="auc_mean", metric_std="auc_std"):
    lines = [f"\n{title}", f"{'Method':22s}{'AUC':>16s}{'PR-AUC':>16s}{'Precision@K':>16s}{'Recall@K':>16s}"]
    for method in EXPERIMENTS:
        a = agg[method]
        def fmt(m, s, pct=False):
            if m is None:
                return "n/a"
            return f"{m:.1%}+/-{s:.1%}" if pct else f"{m:.3f}+/-{s:.3f}"
        lines.append(
            f"{method:22s}{fmt(a['auc_mean'], a['auc_std']):>16s}{fmt(a['pr_auc_mean'], a['pr_auc_std']):>16s}"
            f"{fmt(a['precision_at_k_mean'], a['precision_at_k_std'], True):>16s}"
            f"{fmt(a['recall_at_k_mean'], a['recall_at_k_std'], True):>16s}"
        )
    return "\n".join(lines)


def format_slice_table(title, keys, agg_by_key, metric="pr_auc_mean", metric_std="pr_auc_std"):
    lines = [f"\n{title} (metric: PR-AUC, mean +/- std across seeds)",
             f"{'':16s}" + "".join(f"{m[:16]:>19s}" for m in EXPERIMENTS)]
    for key in keys:
        row = f"{str(key):16s}"
        for method in EXPERIMENTS:
            a = agg_by_key[key][method]
            m, s = a.get(metric), a.get(metric_std)
            cell = f"{m:.1%}+/-{s:.1%}" if m is not None else "n/a"
            row += f"{cell:>19s}"
        lines.append(row)
    return "\n".join(lines)


def format_contribution(contrib):
    lines = ["\nContribution analysis (paired per-seed deltas, mean +/- std across 5 seeds)"]
    lines.append("\nGNN vs Isolation Forest (GNN - IF, on raw features):")
    for metric, d in contrib["gnn_vs_if"].items():
        if d["mean"] is None:
            continue
        sign = "AUC/PR-AUC/etc are on 0-1 scale" if False else ""
        lines.append(
            f"  delta_{metric:16s}: {d['mean']:+.3f} +/- {d['std']:.3f}  "
            f"({d['n_positive_seeds']}/{d['n_seeds']} seeds favor GNN)"
        )
    lines.append("\nGNN + IF vs GNN alone (GNN+IF - GNN):")
    for metric, d in contrib["gnn_plus_if_vs_gnn"].items():
        if d["mean"] is None:
            continue
        lines.append(
            f"  delta_{metric:16s}: {d['mean']:+.3f} +/- {d['std']:.3f}  "
            f"({d['n_positive_seeds']}/{d['n_seeds']} seeds favor GNN+IF)"
        )
    lines.append(
        "\nNote: 5 seeds is a reproducibility estimate (mean +/- std, and how many of the 5\n"
        "agreed on direction), not a formal statistical significance test. No p-value is\n"
        "reported - claiming significance from n=5 would not be honest."
    )
    return "\n".join(lines)


def make_figure(agg_by_tier, out_path):
    tiers = list(TIERS.keys())
    x = np.arange(len(tiers))
    width = 0.2
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, method in enumerate(EXPERIMENTS):
        means = [agg_by_tier[t][method]["pr_auc_mean"] or 0.0 for t in tiers]
        stds = [agg_by_tier[t][method]["pr_auc_std"] or 0.0 for t in tiers]
        ax.bar(x + (i - 1.5) * width, means, width, yerr=stds, capsize=3, label=method)

    ax.set_xticks(x)
    ax.set_xticklabels([t.title() for t in tiers])
    ax.set_ylabel("PR-AUC (mean +/- std, 5 seeds)")
    ax.set_title("Which detector works as fraud becomes less obvious?")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


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

    per_seed = []
    for seed in args.seeds:
        t0 = time.time()
        r = run_one_seed_ablation(report.transactions, seed, args.rings_per_type_tier,
                                   args.epochs, args.hidden_dim, args.embed_dim, args.lr)
        elapsed = time.time() - t0
        print(f"seed {seed}: loss={r['final_loss']:.3f} "
              f"Rules={r['overall']['BankForensIQ Rules']['auc']:.3f} "
              f"IF(raw)={r['overall']['Isolation Forest']['auc']:.3f} "
              f"GNN={r['overall']['GNN']['auc']:.3f} "
              f"GNN+IF={r['overall']['GNN + IF']['auc']:.3f} ({elapsed:.1f}s)")
        per_seed.append(r)

    agg_overall = aggregate(per_seed, lambda r: r["overall"])
    agg_by_type = {t: aggregate(per_seed, lambda r, t=t: r["by_type"][t]) for t in TYPES}
    agg_by_tier = {tier: aggregate(per_seed, lambda r, tier=tier: r["by_tier"][tier]) for tier in TIERS}
    contrib = contribution_analysis(per_seed)

    overall_table = format_table("B. OVERALL RESULTS (5-seed mean +/- std)", agg_overall)
    tier_table = format_slice_table("C. PER-TIER RESULTS", list(TIERS.keys()), agg_by_tier)
    type_table = format_slice_table("D. PER-FRAUD-TYPE RESULTS", TYPES, agg_by_type)
    contrib_text = format_contribution(contrib)

    fig_path = out_dir / "ablation_pr_auc_by_tier.png"
    make_figure(agg_by_tier, fig_path)

    setup = (
        "A. EXPERIMENTAL SETUP\n" + "=" * 60 + "\n"
        f"Benchmark: 150 fraud rings (5 types x 3 tiers x 10), unchanged from v3/v5.\n"
        f"Seeds: {args.seeds}. Epochs: {args.epochs}. GNN: DirectionalGCNAutoencoder\n"
        f"(hidden={args.hidden_dim}, embed={args.embed_dim}, lr={args.lr}), unchanged.\n"
        f"Nodes (seed {args.seeds[0]}): {per_seed[0]['n_nodes']}. "
        f"Excel adapter: {EXCEL_ADAPTER_SOURCE}.\n\n"
        "Four experiments, all scored on the identical graph per seed:\n"
        "  A. GNN only              - reconstruction_scores(), unchanged\n"
        "  B. Isolation Forest only - on RAW pre-GNN features X (corrected/independent\n"
        "                             this round - see src/ablation.py docstring; NOT\n"
        "                             the same IF score used inside experiment C)\n"
        "  C. GNN + Isolation Forest - the EXISTING combination, unchanged:\n"
        "                             0.5*rank(gnn_score) + 0.5*rank(IF-on-Z), where\n"
        "                             IF-on-Z uses the GNN's own embeddings, not X\n"
        "  D. BankForensIQ Transaction Rules - v5-corrected reference baseline, unchanged\n"
    )

    limitations = (
        "\nG. LIMITATIONS\n" + "=" * 60 + "\n"
        "- Experiment C's internal IF component is NOT the same computation as\n"
        "  Experiment B (B uses raw features X; C's internal IF uses GNN embeddings Z).\n"
        "  This is intentional and necessary for B to be a genuine ablation baseline, but\n"
        "  it means C is not simply 'A plus B' - it's 'A plus a different, GNN-derived\n"
        "  IF variant'. Comparing C against (A+B) as if additive would be incorrect.\n"
        "- 5 seeds is a reproducibility estimate, not a significance test (see section E/F).\n"
        "- All limitations from v5's STATE.md still apply unchanged (account-node gap,\n"
        "  reconstructed excel_adapter.py/transaction_schema.py, RTGS, PDFs, etc.) -\n"
        "  not re-litigated here since nothing in this round touched them.\n"
        "- These results evaluate recovery of synthetically injected fraud structures.\n"
        "  They are not measurements of confirmed real-world fraud detection because the\n"
        "  real dataset has no fraud ground-truth labels.\n"
    )

    interpretation = (
        "\nH. INTERPRETATION\n" + "=" * 60 + "\n"
        "Per the task's explicit instruction, this does NOT conclude 'GNN is better\n"
        "because AUC is higher.' What the sliced results actually show is reported in\n"
        "the numbers above - read the per-tier and per-type tables directly rather than\n"
        "the single overall AUC line before drawing conclusions.\n"
    )

    report_text = (
        setup + overall_table + "\n\nE. GNN vs ISOLATION FOREST, F. GNN vs GNN+IF\n" + "=" * 60 +
        contrib_text + tier_table + type_table + limitations + interpretation
    )
    print("\n" + report_text)

    with open(out_dir / "ablation_report.txt", "w") as f:
        f.write(report_text)

    with open(out_dir / "ablation_results.json", "w") as f:
        json.dump(dict(
            config=dict(seeds=args.seeds, rings_per_type_tier=args.rings_per_type_tier,
                        epochs=args.epochs, hidden_dim=args.hidden_dim, embed_dim=args.embed_dim,
                        lr=args.lr, total_transactions=len(report.transactions)),
            per_seed=per_seed,
            aggregate_overall=agg_overall, aggregate_by_type=agg_by_type, aggregate_by_tier=agg_by_tier,
            contribution_analysis=contrib,
        ), f, indent=2, default=str)

    print(f"\nWrote ablation_report.txt, ablation_results.json, {fig_path.name} to {out_dir}")


if __name__ == "__main__":
    main()
