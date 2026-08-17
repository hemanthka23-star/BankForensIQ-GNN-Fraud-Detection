"""
Held-out fraud-type generalization - leave-one-fraud-type-out testing.

────────────────────────────────────────────────────────────────────
Inspection of the current training procedure (task's mandatory first
step) - read from the actual code, not assumed
────────────────────────────────────────────────────────────────────
graph_builder.directed_edge_index() builds the GNN's positive training
set from `set(g.edges())` with NO distinction between real transaction
edges and injected synthetic edges, and no distinction between fraud
types - every edge in whatever graph is passed to train_directional_gae
becomes something the autoencoder is trained to reconstruct. Confirmed
by reading the source, not assumed:

    pairs = set(g.edges())   # graph_builder.py:205 - ALL edges, no filter

Consequence: if a single 5-type graph were built and then a fraud
type's LABELS were merely hidden from evaluation, the GNN would still
have been trained to reconstruct that exact fraud type's edges - not a
valid held-out test. This is exactly the contamination task step 2
warns about, and it is real in this codebase, not hypothetical.

However, the model's weight matrices are shaped by feature dimension
and hidden/embed size only (n_features x hidden, hidden x embed, embed
x embed - see gnn_model.py's DirectionalGCNAutoencoder.__init__) -
NONE of them depend on the node count N. This means the architecture
is inductive: a model trained on one graph can be validly applied,
forward-pass only, to score a DIFFERENT graph with a different node
set - which is exactly what a genuine held-out test needs. This is why
the "critical stopping rule" (task step 15) is NOT invoked - the
architecture supports a valid experiment via a real train/test graph
split, no architectural change needed.

────────────────────────────────────────────────────────────────────
The design (task steps 1-3)
────────────────────────────────────────────────────────────────────
TRAIN graph:  build_multigraph(txns) [same real background, always]
              + inject_synthetic_rings(..., types=<4 non-held-out types>,
                seed=seed)
              GNN is trained ONLY on this graph - train_directional_gae
              never sees the held-out type's structures.

TEST graph:   a SEPARATE build_multigraph(txns) call [same real
              background transactions, rebuilt independently - it's
              the same deterministic real data either way, by design,
              per task step 3: "same underlying real/background
              transactions but independently generated synthetic
              injections"]
              + inject_synthetic_rings(..., types=<held-out type only>,
                seed=a DIFFERENT seed, offset from the train seed)
              then every synthetic node this produces is relabeled
              SYN: -> SYNTEST: (a pure string rename via
              nx.relabel_nodes - ring topology/amounts/timing
              untouched) so train and test synthetic IDs cannot
              collide even in principle, not just "shouldn't".
              The trained model only ever sees this graph via a single
              forward() call (inference) - train_directional_gae is
              never called on it.

Feature normalization note: compute_node_features (unchanged) z-scores
using whichever graph's own population it's given. Train and test
graphs are recomputed independently here rather than sharing train's
mean/std, since the real background (4,861 nodes) dominates the
population in both graphs (at most ~150 synthetic nodes added either
way), so normalization statistics are close between them in practice;
documented as a deliberate simplification, not hidden, and avoids
modifying features.py this round.
"""

import sys
import warnings
from pathlib import Path
from typing import Dict, List

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import networkx as nx
import numpy as np

from graph_builder import build_multigraph, build_adjacency, build_directional_channels, directed_edge_index
from labeling import inject_synthetic_rings, TYPES, TIERS, SYNTHETIC_PREFIX
from features import compute_node_features
from gnn_model import train_directional_gae, DirectionalGCNAutoencoder, normalize_symmetric
from evaluate import reconstruction_scores, isolation_scores
from metrics import compute_metrics

TEST_SEED_OFFSET = 10_007  # arbitrary, large, prime-ish - just needs to
                            # decorrelate the test injection's RNG stream
                            # from the train injection's
TEST_PREFIX = "SYNTEST:"


def build_train_graph(txns, held_out_type: str, rings_per_type_tier: int, seed: int):
    train_types = [t for t in TYPES if t != held_out_type]
    g, _ = build_multigraph(txns)
    g, ring_records = inject_synthetic_rings(
        g, rings_per_type_tier=rings_per_type_tier, seed=seed, types=train_types
    )
    return g, ring_records


def build_test_graph(txns, held_out_type: str, rings_per_type_tier: int, seed: int):
    test_seed = seed + TEST_SEED_OFFSET
    g, _ = build_multigraph(txns)  # same deterministic real background, fresh object
    g, ring_records = inject_synthetic_rings(
        g, rings_per_type_tier=rings_per_type_tier, seed=test_seed, types=[held_out_type]
    )

    # Relabel every synthetic node SYN: -> SYNTEST: - guarantees zero
    # string-level ID overlap with ANY train graph's synthetic nodes,
    # not just "different seed should make collision unlikely".
    mapping = {n: n.replace(SYNTHETIC_PREFIX, TEST_PREFIX) for n in g.nodes() if n.startswith(SYNTHETIC_PREFIX)}
    g = nx.relabel_nodes(g, mapping, copy=True)

    for record in ring_records:
        record["nodes"] = [mapping.get(n, n) for n in record["nodes"]]
        record["core_nodes"] = [mapping.get(n, n) for n in record["core_nodes"]]

    return g, ring_records, test_seed


def train_on_graph(g_train, seed, epochs=200, hidden_dim=64, embed_dim=32, lr=0.02):
    X_train, node_list_train, _ = compute_node_features(g_train)
    adj_sym_train, node_list_check = build_adjacency(g_train)
    assert node_list_train == node_list_check
    channels_train = build_directional_channels(g_train, node_list_train)
    pos_i, pos_j = directed_edge_index(g_train, node_list_train)
    channels_train["_pos_i"], channels_train["_pos_j"] = pos_i, pos_j

    model, Z_train, cache_train, losses = train_directional_gae(
        X_train, channels_train, adj_sym_train, epochs=epochs, lr=lr,
        hidden_dim=hidden_dim, embed_dim=embed_dim, seed=seed, verbose=False,
    )
    return model, node_list_train, losses


def score_test_graph(model: DirectionalGCNAutoencoder, g_test: nx.MultiDiGraph, seed: int):
    """
    Inference only - train_directional_gae is never called here. Scores
    the test graph using the already-trained model's forward() (a plain
    encoder pass + decoder score, no weight updates).
    """
    X_test, node_list_test, _ = compute_node_features(g_test)
    adj_sym_test, node_list_check = build_adjacency(g_test)
    assert node_list_test == node_list_check
    channels_test = build_directional_channels(g_test, node_list_test)
    A_sym_test = normalize_symmetric(adj_sym_test)

    Z_test, cache_test = model.forward(X_test, channels_test, A_sym_test)

    gnn_score = reconstruction_scores(model, cache_test, g_test, node_list_test)
    if_score = isolation_scores(X_test, seed=seed)  # independent IF, on raw test features

    return gnn_score, if_score, node_list_test


def _labels_for_test_graph(node_list_test, ring_records_test, slice_tier=None):
    """Returns (labels, keep_mask). When slicing by tier, synthetic
    nodes from OTHER tiers are excluded via keep_mask (not counted as
    negatives for a tier they don't belong to) - same pattern as
    run_comparison.py/ablation.py's slice masking."""
    nodes_in_slice = set()
    all_synth_nodes = set()
    for r in ring_records_test:
        all_synth_nodes.update(r["nodes"])
        if slice_tier is None or r["tier"] == slice_tier:
            nodes_in_slice.update(r["nodes"])
    labels = np.array([n in nodes_in_slice for n in node_list_test])
    other_tier_synth = np.array([(n in all_synth_nodes) and (n not in nodes_in_slice) for n in node_list_test])
    return labels, ~other_tier_synth


def run_one_heldout_experiment(txns, held_out_type: str, seed: int,
                                rings_per_type_tier=10, epochs=200,
                                hidden_dim=64, embed_dim=32, lr=0.02):
    g_train, ring_records_train = build_train_graph(txns, held_out_type, rings_per_type_tier, seed)
    g_test, ring_records_test, test_seed = build_test_graph(txns, held_out_type, rings_per_type_tier, seed)

    model, node_list_train, losses = train_on_graph(
        g_train, seed, epochs, hidden_dim, embed_dim, lr
    )
    gnn_score, if_score, node_list_test = score_test_graph(model, g_test, seed)

    labels_all, keep_all = _labels_for_test_graph(node_list_test, ring_records_test)
    overall_gnn = compute_metrics(gnn_score[keep_all], labels_all[keep_all])
    overall_if = compute_metrics(if_score[keep_all], labels_all[keep_all])

    by_tier_gnn, by_tier_if = {}, {}
    for tier in TIERS:
        labels_t, keep_t = _labels_for_test_graph(node_list_test, ring_records_test, slice_tier=tier)
        by_tier_gnn[tier] = compute_metrics(gnn_score[keep_t], labels_t[keep_t])
        by_tier_if[tier] = compute_metrics(if_score[keep_t], labels_t[keep_t])

    train_synth_nodes = set()
    for r in ring_records_train:
        train_synth_nodes.update(r["nodes"])
    test_synth_nodes = set()
    for r in ring_records_test:
        test_synth_nodes.update(r["nodes"])

    return dict(
        held_out_type=held_out_type,
        seed=seed, test_seed=test_seed,
        n_train_nodes=g_train.number_of_nodes(), n_test_nodes=g_test.number_of_nodes(),
        n_train_rings=len(ring_records_train), n_test_rings=len(ring_records_test),
        final_train_loss=float(losses[-1]),
        train_synth_node_count=len(train_synth_nodes),
        test_synth_node_count=len(test_synth_nodes),
        train_test_node_id_overlap=len(train_synth_nodes & test_synth_nodes),  # must be 0
        overall_gnn=overall_gnn, overall_if=overall_if,
        by_tier_gnn=by_tier_gnn, by_tier_if=by_tier_if,
    )
