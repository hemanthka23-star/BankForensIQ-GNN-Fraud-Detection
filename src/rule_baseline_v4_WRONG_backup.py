"""
Deterministic rule-based baseline - reproduces the ACTUAL BankForensIQ
logic found by inspecting backend/services/unified_fraud_engine.py and
risk_engine.py (see gnn_fraud_intelligence/README.md "v4 update" for the
full discrepancy report). Two of the four requested signals are real,
faithfully-ported logic; two are explicit stubs, documented below and
never contributing invented detection logic (they return exactly 0.0
for every node).

burst_score()            REAL. Ports RAPID_TRANSACTION (risk_engine.py):
                          >= RAPID_BURST_THRESHOLD (4) transactions by
                          this node within a RAPID_WINDOW_MINUTES (10)
                          sliding window, with total window amount
                          > RAPID_AMOUNT_FLOOR (Rs 10,000).
smurfing_proximity_score() REAL. Ports is_near_threshold + the "Smurfing"
                          episode label (unified_fraud_engine.py): the
                          fraction of this node's transactions with
                          amount in [9000, 9999] (the exact real band -
                          staying just under a Rs 10,000-style
                          reporting threshold). The real system labels
                          an episode "Smurfing" when this fraction
                          exceeds 0.50; this returns the fraction
                          itself (continuous, for ranking) rather than
                          thresholding to a binary label.
circular_return_score()  STUB. No circular/multi-hop-return detection
                          exists anywhere in the codebase - confirmed by
                          exhaustive keyword + class/function-name
                          search. Always returns 0.0.
money_flow_score()       STUB. No counterparty-network "money flow"
                          concept exists. The closest real analog
                          (single-account balance-drain/zeroing) needs
                          per-transaction `balance`, which this
                          project's graph_builder.py does not currently
                          carry onto edges - reproducing it would need a
                          graph-schema change, deferred per this round's
                          explicit "don't modify the GNN/graph pipeline
                          unless a bug is found" scope. Always returns
                          0.0.

Important adaptation, stated explicitly rather than left implicit: the
real rules are account-centric (computed from one account's own full
statement). This graph also has counterparty nodes, which have no
"own statement" - only the edges that touch them from our accounts'
statements. Both real functions below are applied node-by-node using
whatever edges touch that node (as either endpoint), regardless of
node_type, since that's the only data available for counterparty nodes
and it keeps every node scored as required. This is a genuine
adaptation of an account-centric system to a graph, not a hidden one.

combined_rule_score() uses max(burst, smurfing) - not the real system's
actual combination (an additive RULE_POINTS scheme feeding a calibrated
ML+rule FusionLayer), because that requires trained ML models scoring
transaction-level features that don't have a graph-node analog. Per the
task's explicit permission to "prefer maximum... document the exact
choice" for a first baseline, max is used here. Since both stubs
contribute a constant 0.0, they can never change the combined score -
this baseline is exactly as strong as the two real signals, no more.
"""

from collections import defaultdict
from typing import Dict, List

import networkx as nx
import numpy as np

RAPID_WINDOW_MINUTES = 10
RAPID_BURST_THRESHOLD = 4
RAPID_AMOUNT_FLOOR = 10_000.0

SMURF_AMOUNT_LOW = 9000.0
SMURF_AMOUNT_HIGH = 9999.0


def _node_edges(g: nx.MultiDiGraph, node: str):
    """All (amount, parsed_date) pairs for transactions touching this
    node as either sender or receiver."""
    out = [(d["amount"], d.get("parsed_date")) for _, _, d in g.out_edges(node, data=True)]
    inn = [(d["amount"], d.get("parsed_date")) for _, _, d in g.in_edges(node, data=True)]
    return out + inn


def burst_score(g: nx.MultiDiGraph, node_list: List[str]) -> np.ndarray:
    scores = np.zeros(len(node_list), dtype=np.float32)
    window = pd_timedelta_minutes(RAPID_WINDOW_MINUTES)

    for i, node in enumerate(node_list):
        events = [(dt, amt) for amt, dt in _node_edges(g, node) if dt is not None]
        if len(events) < RAPID_BURST_THRESHOLD:
            continue
        events.sort()
        timestamps = [e[0] for e in events]
        amounts = [e[1] for e in events]

        best_count = 0
        n = len(events)
        left = 0
        window_amount = 0.0
        for right in range(n):
            window_amount += amounts[right]
            while (timestamps[right] - timestamps[left]) > window:
                window_amount -= amounts[left]
                left += 1
            count = right - left + 1
            if count >= RAPID_BURST_THRESHOLD and window_amount > RAPID_AMOUNT_FLOOR:
                best_count = max(best_count, count)

        if best_count >= RAPID_BURST_THRESHOLD:
            # continuous score: how far past the threshold, saturating at 1.0
            scores[i] = min(1.0, (best_count - RAPID_BURST_THRESHOLD + 1) / 6.0)

    return scores


def pd_timedelta_minutes(minutes):
    from datetime import timedelta
    return timedelta(minutes=minutes)


def smurfing_proximity_score(g: nx.MultiDiGraph, node_list: List[str]) -> np.ndarray:
    scores = np.zeros(len(node_list), dtype=np.float32)

    for i, node in enumerate(node_list):
        edges = _node_edges(g, node)
        if not edges:
            continue
        amounts = [amt for amt, _ in edges]
        near = sum(1 for a in amounts if SMURF_AMOUNT_LOW <= a <= SMURF_AMOUNT_HIGH)
        scores[i] = near / len(amounts)

    return scores


def circular_return_score(g: nx.MultiDiGraph, node_list: List[str]) -> np.ndarray:
    """STUB - see module docstring. No such logic exists in BankForensIQ."""
    return np.zeros(len(node_list), dtype=np.float32)


def money_flow_score(g: nx.MultiDiGraph, node_list: List[str]) -> np.ndarray:
    """STUB - see module docstring. Real analog exists but isn't portable
    without a graph-schema change deferred this round."""
    return np.zeros(len(node_list), dtype=np.float32)


def combined_rule_score(g: nx.MultiDiGraph, node_list: List[str]) -> Dict[str, np.ndarray]:
    """
    Returns all component scores plus the combined score, so each can be
    evaluated independently (per the task's step 3: "do not immediately
    combine them") as well as together.
    """
    burst = burst_score(g, node_list)
    smurf = smurfing_proximity_score(g, node_list)
    circular = circular_return_score(g, node_list)
    money_flow = money_flow_score(g, node_list)

    combined = np.maximum(np.maximum(burst, smurf), np.maximum(circular, money_flow))

    return dict(
        burst=burst,
        smurfing_proximity=smurf,
        circular_return=circular,
        money_flow=money_flow,
        combined=combined,
    )


if __name__ == "__main__":
    import sys
    import pickle
    import warnings
    sys.path.insert(0, ".")
    warnings.filterwarnings("ignore")

    from graph_builder import build_multigraph
    from labeling import inject_synthetic_rings
    from ingest import ingest_directory

    print("Smoke test: ingesting + injecting + scoring on a small run...")
    report = ingest_directory(sys.argv[1] if len(sys.argv) > 1 else "/home/claude/dataset/Bank-statements-dataset", verbose=False)
    g, _ = build_multigraph(report.transactions)
    g, ring_records = inject_synthetic_rings(g, rings_per_type_tier=2, seed=42)
    node_list = list(g.nodes())

    scores = combined_rule_score(g, node_list)
    for name, arr in scores.items():
        print(f"{name:20s} min={arr.min():.3f} max={arr.max():.3f} mean={arr.mean():.4f} nonzero={np.count_nonzero(arr)}")

    assert scores["combined"].min() >= 0.0 and scores["combined"].max() <= 1.0
    assert len(scores["combined"]) == len(node_list)
    print("\nOK: scores in [0,1], every node scored")
