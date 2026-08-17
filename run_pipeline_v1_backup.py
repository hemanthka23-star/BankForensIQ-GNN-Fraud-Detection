"""
Run the full GNN fraud-graph pipeline end-to-end against a real dataset
directory (matches the structure of Bank-statements-dataset.zip: any mix
of .csv / .xlsx / .xls files, nested in subfolders or not).

    python run_pipeline.py --data-dir /path/to/Bank-statements-dataset

Writes to data/processed/:
    node_features.csv        per-node feature table (includes real
                              names/UPI handles/etc - local inspection
                              only, see README "PII handling")
    graph.gpickle             the full transaction MultiDiGraph
    embeddings.npy            trained GCN node embeddings (N x embed_dim)
    anomaly_scores.csv        node_id, anomaly_score, percentile_rank
    evaluation_report.txt     synthetic-ring recovery metrics
    subgraphs/*.png           anonymized visualizations of 3 sample rings
    run_summary.json          aggregate counts only, no PII - safe to
                              share/paste elsewhere
"""

import argparse
import json
import pickle
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import networkx as nx
import numpy as np
import pandas as pd

from ingest import ingest_directory, EXCEL_ADAPTER_SOURCE
from graph_builder import build_multigraph, build_adjacency
from labeling import inject_synthetic_rings
from features import compute_node_features
from gnn_model import train_gae, GCNAutoencoder
from evaluate import (
    reconstruction_scores, isolation_scores, combined_anomaly_score,
    evaluate_recovery, format_report, _rank_normalize,
)
from visualize import render_ring_subgraph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Root folder of bank statements")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--n-rings", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timings = {}

    print(f"[1/7] Ingesting {args.data_dir}  (excel adapter: {EXCEL_ADAPTER_SOURCE})")
    t0 = time.time()
    report = ingest_directory(args.data_dir, verbose=True)
    timings["ingest"] = time.time() - t0
    print(report.summary)

    print("\n[2/7] Building transaction graph")
    t0 = time.time()
    g, graph_report = build_multigraph(report.transactions)
    timings["graph_build"] = time.time() - t0
    print(graph_report.summary)
    print(f"Nodes: {g.number_of_nodes()}  Edges: {g.number_of_edges()}")

    print(f"\n[3/7] Injecting {args.n_rings} synthetic fraud rings (validation harness)")
    t0 = time.time()
    g, ring_records = inject_synthetic_rings(g, n_rings=args.n_rings, seed=args.seed)
    timings["inject"] = time.time() - t0
    print(f"Nodes now: {g.number_of_nodes()}  Edges now: {g.number_of_edges()}")

    print("\n[4/7] Computing node features")
    t0 = time.time()
    X, node_list, feat_df = compute_node_features(g)
    timings["features"] = time.time() - t0
    print(f"Feature matrix: {X.shape}")

    print(f"\n[5/7] Training GCN graph autoencoder ({args.epochs} epochs)")
    t0 = time.time()
    adj, node_list2 = build_adjacency(g)
    assert node_list == node_list2
    model, Z, A_hat, losses = train_gae(
        X, adj, epochs=args.epochs, lr=args.lr,
        hidden_dim=args.hidden_dim, embed_dim=args.embed_dim,
        seed=args.seed, verbose=True,
    )
    timings["train"] = time.time() - t0

    print("\n[6/7] Scoring anomalies and evaluating synthetic-ring recovery")
    t0 = time.time()
    recon = reconstruction_scores(model, Z, g, node_list)
    iso = isolation_scores(Z, seed=args.seed)
    combined = combined_anomaly_score(recon, iso)
    results = evaluate_recovery(combined, node_list, ring_records)
    report_text = format_report(results)
    print(report_text)
    timings["evaluate"] = time.time() - t0

    print("\n[7/7] Writing artifacts")
    t0 = time.time()

    feat_df.to_csv(out_dir / "node_features.csv", index=False)

    with open(out_dir / "graph.gpickle", "wb") as f:
        pickle.dump(g, f)

    np.save(out_dir / "embeddings.npy", Z)

    percentile = _rank_normalize(combined)
    pd.DataFrame({
        "node_id": node_list,
        "is_synthetic": [n.startswith("SYN:") for n in node_list],
        "anomaly_score": combined,
        "percentile_rank": percentile,
    }).sort_values("anomaly_score", ascending=False).to_csv(
        out_dir / "anomaly_scores.csv", index=False
    )

    with open(out_dir / "evaluation_report.txt", "w") as f:
        f.write(report_text + "\n")

    sub_dir = out_dir / "subgraphs"
    sub_dir.mkdir(exist_ok=True)
    chosen = {}
    for r in ring_records:
        chosen.setdefault(r["ring_type"], r)
    subgraph_files = []
    for ring_type, record in chosen.items():
        p = sub_dir / f"ring_{record['ring_id']}_{ring_type}.png"
        render_ring_subgraph(g, node_list, combined, record, str(p))
        subgraph_files.append(str(p))

    timings["artifacts"] = time.time() - t0

    summary = dict(
        excel_adapter_used=EXCEL_ADAPTER_SOURCE,
        files_parsed_ok=len(report.ok_files),
        files_failed=len(report.failed_files),
        failed_files=[dict(name=n, error=e) for n, _, e in report.failed_files],
        files_skipped_unsupported_extension=len(report.skipped_extensions),
        total_transactions=len(report.transactions),
        graph_nodes_real=g.number_of_nodes() - len(
            [n for n in node_list if n.startswith("SYN:")]
        ),
        graph_nodes_synthetic=len([n for n in node_list if n.startswith("SYN:")]),
        graph_edges=g.number_of_edges(),
        synthetic_rings_injected=len(ring_records),
        final_train_loss=float(losses[-1]),
        initial_train_loss=float(losses[0]),
        evaluation=results,
        timings_seconds={k: round(v, 2) for k, v in timings.items()},
    )
    with open(out_dir / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nDone. Artifacts written to {out_dir.resolve()}")
    print(f"Total wall time: {sum(timings.values()):.1f}s")


if __name__ == "__main__":
    main()
