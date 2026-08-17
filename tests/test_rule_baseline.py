"""
Tests for the rule baseline / comparison harness, per the task's
explicit requirements: score bounds, full node coverage, consistent
node ordering, matching K, and - most important - that the rule
detector is completely independent of the GNN and never sees synthetic
labels while scoring.

Run with:  python -m pytest tests/ -v
       or:  python tests/test_rule_baseline.py
"""

import sys
import inspect
import warnings
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import networkx as nx
import numpy as np

from ingest import ingest_directory
from graph_builder import build_multigraph
from labeling import inject_synthetic_rings
import rule_baseline
from rule_baseline import combined_rule_score
from metrics import compute_metrics


def _small_graph(seed=42):
    report = ingest_directory("/home/claude/dataset/Bank-statements-dataset", verbose=False)
    g, _ = build_multigraph(report.transactions)
    g, ring_records = inject_synthetic_rings(g, rings_per_type_tier=2, seed=seed)
    node_list = list(g.nodes())
    return g, node_list, ring_records


def test_rule_score_bounds():
    g, node_list, _ = _small_graph()
    scores = combined_rule_score(g, node_list)
    for name, arr in scores.items():
        assert arr.min() >= 0.0, f"{name} has a score below 0"
        assert arr.max() <= 1.0, f"{name} has a score above 1"
    print("PASS: all rule scores in [0, 1]")


def test_every_node_scored():
    g, node_list, _ = _small_graph()
    scores = combined_rule_score(g, node_list)
    for name, arr in scores.items():
        assert len(arr) == len(node_list), f"{name} length {len(arr)} != {len(node_list)} nodes"
        assert np.isfinite(arr).all(), f"{name} has non-finite values"
    print("PASS: every node receives a score, no NaN/Inf")


def test_consistent_node_ordering():
    """The array index for scores must line up with node_list in the
    same order every time - this is what lets rule/GNN/label arrays be
    compared element-wise without a join."""
    g, node_list, ring_records = _small_graph(seed=42)
    scores_a = combined_rule_score(g, node_list)
    scores_b = combined_rule_score(g, node_list)
    assert np.array_equal(scores_a["combined"], scores_b["combined"]), \
        "same graph + same node_list must give identical, deterministically-ordered scores"
    print("PASS: node ordering / determinism consistent across repeated calls")


def test_labels_match_synthetic_nodes_exactly():
    g, node_list, ring_records = _small_graph()
    all_synthetic = set()
    for r in ring_records:
        all_synthetic.update(r["nodes"])

    labels = np.array([n.startswith("SYN:") for n in node_list])
    labels_from_records = np.array([n in all_synthetic for n in node_list])
    assert np.array_equal(labels, labels_from_records), \
        "SYN: prefix check and ring_records node membership must agree exactly"
    print(f"PASS: {labels.sum()} synthetic nodes, labels match ring_records exactly")


def test_k_is_explicit_and_consistent():
    g, node_list, ring_records = _small_graph()
    labels = np.array([n.startswith("SYN:") for n in node_list])
    scores = combined_rule_score(g, node_list)["combined"]

    m1 = compute_metrics(scores, labels)  # default k = n_positive
    m2 = compute_metrics(scores, labels, k=labels.sum())
    assert m1["k"] == m2["k"] == int(labels.sum()), "default K must equal n_positive and be explicit in output"
    print(f"PASS: K={m1['k']} explicit and consistent (== n_positive)")


def test_no_synthetic_label_passed_into_rule_detector():
    """Static check: none of the real rule-scoring functions take a
    labels/is_synthetic argument - they only ever see the graph and
    node_list, so there's no code path for a label to leak in."""
    for fn in (rule_baseline.burst_score, rule_baseline.high_value_score,
               rule_baseline.spending_spike_score, rule_baseline.repeated_transaction_score,
               rule_baseline.smurfing_episode_score, rule_baseline.circular_return_score,
               rule_baseline.balance_drop_score, rule_baseline.combined_rule_score):
        params = list(inspect.signature(fn).parameters.keys())
        forbidden = [p for p in params if "label" in p.lower() or "synthetic" in p.lower() or "fraud" in p.lower()]
        assert not forbidden, f"{fn.__name__} has a suspicious parameter: {forbidden}"
    print("PASS: no rule-scoring function accepts a label/synthetic/fraud argument")


def test_rule_detector_does_not_import_gnn():
    """Static check: rule_baseline.py must not import anything from
    gnn_model.py or evaluate.py's GNN-specific functions - keeps the
    baseline provably independent of the GNN's output."""
    src = Path(__file__).resolve().parent.parent / "src" / "rule_baseline.py"
    text = src.read_text()
    assert "gnn_model" not in text, "rule_baseline.py must not import gnn_model"
    assert "reconstruction_scores" not in text, "rule_baseline.py must not use the GNN's reconstruction score"
    print("PASS: rule_baseline.py has no GNN dependency (source-level check)")


def _mini_graph_with_timestamps(node, timestamps, direction="OUT"):
    """Builds a minimal graph where `node` has one transaction per
    timestamp given, all with a fixed innocuous amount (so only the
    burst logic, not amount-based rules, can fire)."""
    g = nx.MultiDiGraph()
    g.add_node(node, node_type="account")
    for i, dt in enumerate(timestamps):
        cp = f"CP{i}"
        g.add_node(cp, node_type="counterparty")
        if direction == "OUT":
            g.add_edge(node, cp, amount=500.0, parsed_date=dt)
        else:
            g.add_edge(cp, node, amount=500.0, parsed_date=dt)
    return g


def test_rapid_transaction_4_in_10min_does_not_trigger():
    """Exact scenario from the task: 4 transactions within the 10-minute
    window must NOT trigger (real threshold is 5, corrected from v4's
    incorrect 4)."""
    base = datetime(2025, 1, 1, 12, 0, 0)
    timestamps = [base, base + timedelta(minutes=2), base + timedelta(minutes=4), base + timedelta(minutes=6)]
    g = _mini_graph_with_timestamps("A", timestamps)
    score = rule_baseline.burst_score(g, ["A"])[0]
    assert score == 0.0, f"4 transactions in 10 min should NOT trigger burst, got score={score}"
    print("PASS: 4 transactions in 10 minutes does NOT trigger (threshold is 5)")


def test_rapid_transaction_5_in_10min_triggers():
    """Exact scenario from the task: 5 transactions within the window
    SHOULD trigger."""
    base = datetime(2025, 1, 1, 12, 0, 0)
    timestamps = [base + timedelta(minutes=m) for m in (0, 2, 4, 6, 8)]
    g = _mini_graph_with_timestamps("A", timestamps)
    score = rule_baseline.burst_score(g, ["A"])[0]
    assert score == 1.0, f"5 transactions within the window SHOULD trigger burst, got score={score}"
    print("PASS: 5 transactions in 10 minutes DOES trigger")


def test_rapid_transaction_5_outside_window_does_not_trigger():
    """Exact scenario from the task: 5 transactions spread OUTSIDE the
    10-minute window must NOT trigger."""
    base = datetime(2025, 1, 1, 12, 0, 0)
    timestamps = [base + timedelta(hours=h) for h in (0, 2, 4, 6, 8)]  # hours apart, not minutes
    g = _mini_graph_with_timestamps("A", timestamps)
    score = rule_baseline.burst_score(g, ["A"])[0]
    assert score == 0.0, f"5 transactions hours apart should NOT trigger burst, got score={score}"
    print("PASS: 5 transactions spread hours apart does NOT trigger")


def test_rapid_transaction_ignores_direction():
    """The real rule does not filter by debit/credit - a mix of IN and
    OUT transactions within the window must still trigger, since v4's
    bug was incorrectly filtering to debit-only."""
    base = datetime(2025, 1, 1, 12, 0, 0)
    g = nx.MultiDiGraph()
    g.add_node("A", node_type="account")
    for i, (m, direction) in enumerate([(0, "OUT"), (2, "IN"), (4, "OUT"), (6, "IN"), (8, "OUT")]):
        cp = f"CP{i}"
        g.add_node(cp, node_type="counterparty")
        dt = base + timedelta(minutes=m)
        if direction == "OUT":
            g.add_edge("A", cp, amount=500.0, parsed_date=dt)
        else:
            g.add_edge(cp, "A", amount=500.0, parsed_date=dt)
    score = rule_baseline.burst_score(g, ["A"])[0]
    assert score == 1.0, "5 mixed-direction transactions in window SHOULD trigger (no debit-only filter)"
    print("PASS: burst correctly ignores debit/credit direction, matching the real rule")


def test_smurfing_amount_boundaries():
    """9000 and 9999 are inclusive boundaries; 8999 and 10000 are not
    'near threshold' - and a qualifying episode (>50% near-threshold)
    must trigger the smurfing score."""
    base = datetime(2025, 1, 1, 12, 0, 0)
    g = nx.MultiDiGraph()
    g.add_node("A", node_type="account")
    # 3 near-threshold + 1 not, all within 24h -> 75% > 50% -> should trigger
    amounts = [9000.0, 9999.0, 9500.0, 20000.0]
    for i, amt in enumerate(amounts):
        cp = f"CP{i}"
        g.add_node(cp, node_type="counterparty")
        g.add_edge("A", cp, amount=amt, parsed_date=base + timedelta(hours=i))
    score = rule_baseline.smurfing_episode_score(g, ["A"])[0]
    assert score == 1.0, f"3/4 (75%) near-threshold in one episode should trigger smurfing, got {score}"
    print("PASS: smurfing episode score triggers when >50% of an episode is in [9000, 9999]")

    # boundary check: 8999 and 10000 are NOT near-threshold
    g2 = nx.MultiDiGraph()
    g2.add_node("B", node_type="account")
    for i, amt in enumerate([8999.0, 10000.0, 8999.0, 10000.0]):
        cp = f"CP{i}"
        g2.add_node(cp, node_type="counterparty")
        g2.add_edge("B", cp, amount=amt, parsed_date=base + timedelta(hours=i))
    score2 = rule_baseline.smurfing_episode_score(g2, ["B"])[0]
    assert score2 == 0.0, f"amounts just outside [9000,9999] should not count as near-threshold, got {score2}"
    print("PASS: 8999/10000 correctly excluded from the near-threshold band (exclusive boundaries)")


ALL_TESTS = [
    test_rule_score_bounds,
    test_every_node_scored,
    test_consistent_node_ordering,
    test_labels_match_synthetic_nodes_exactly,
    test_k_is_explicit_and_consistent,
    test_no_synthetic_label_passed_into_rule_detector,
    test_rule_detector_does_not_import_gnn,
    test_rapid_transaction_4_in_10min_does_not_trigger,
    test_rapid_transaction_5_in_10min_triggers,
    test_rapid_transaction_5_outside_window_does_not_trigger,
    test_rapid_transaction_ignores_direction,
    test_smurfing_amount_boundaries,
]

if __name__ == "__main__":
    failures = 0
    for test in ALL_TESTS:
        try:
            test()
        except AssertionError as e:
            failures += 1
            print(f"FAIL: {test.__name__}: {e}")
    print(f"\n{len(ALL_TESTS) - failures}/{len(ALL_TESTS)} tests passed")
    if failures:
        raise SystemExit(1)
