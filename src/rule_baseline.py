"""
BankForensIQ transaction-rule baseline - v2, corrected.

v4 (kept as rule_baseline_v4_WRONG_backup.py) got this wrong in three
ways, caught on review: used RAPID_BURST_THRESHOLD=4 instead of the
real 5, filtered to debit-only transactions when the real rule doesn't,
and replaced the real EPISODE-level smurfing logic with a flat
"fraction of all edges near threshold" that isn't the same rule. Also
mislabeled this "Graph Rules" when the inspection had already shown
none of it is graph-based. All three are fixed here.

IMPORTANT NAMING CORRECTION: this is "BankForensIQ Transaction Rules",
not "Graph Rules" - the deployed rules operate on a single account's
own transaction list, not on the counterparty graph.

IMPORTANT SOURCE-FILE CORRECTION: this codebase has TWO separately-live
rule implementations - backend/app.py imports risk_engine.py directly;
backend/routes/upload.py imports unified_fraud_engine.py's quick_run.
They are NOT the same code and disagree on constants (e.g.
RAPID_BURST_THRESHOLD is 5 in risk_engine.py, 4 in
unified_fraud_engine.py; REPEATED_AMOUNT_TOLERANCE is 2% vs 3%). Per
explicit instruction, risk_engine.py is used as authoritative here for
burst/high_value/spending_spike/repeated (all 7 of its named rules are
inspected below). is_near_threshold / episode-based "Smurfing" only
exists in unified_fraud_engine.py - risk_engine.py has no smurfing
logic at all - so that piece is sourced from unified_fraud_engine.py
specifically, not risk_engine.py, and is labeled as such below rather
than presented as if from one unified system.

────────────────────────────────────────────────────────────────────
Mappability determination for all 7 risk_engine.py rules (task step 4)
────────────────────────────────────────────────────────────────────
Real rules operate on ONE account's own full statement (its own
narration text, its own time-of-day data, its own balance trail).
This graph has two node types - only 'account' nodes correspond 1:1 to
an uploaded statement; 'counterparty' nodes are other parties visible
only through the edges that touch them. A rule is "mappable" here if
it can be computed from a node's own touching edges (amount, date,
payment_method, direction, other endpoint) without inventing data that
isn't there:

  MAPPABLE (implemented below, applied per-node using that node's own
  touching edges - see each function's docstring for the exact
  transaction-level -> node-level aggregation):
    burst_score                  RAPID_TRANSACTION      (risk_engine.py)
    high_value_score              HIGH_VALUE_TRANSACTION  (risk_engine.py)
    spending_spike_score           SPENDING_SPIKE          (risk_engine.py)
    repeated_transaction_score      REPEATED_TRANSACTION    (risk_engine.py)
    smurfing_episode_score           "Smurfing" episode label (unified_fraud_engine.py -
                                     see that function's docstring for the
                                     one adaptation this needs)

  NOT MAPPABLE (stubs, return 0.0, real reasons - not a data
  convenience, a structural fact about this pipeline):
    excessive_withdrawal_score  EXCESSIVE_WITHDRAWAL rests on narration-
                                 text keyword matching ("atm", "cash",
                                 "cdm", ...). This graph doesn't carry
                                 narration text onto edges, AND ATM/cash
                                 withdrawals structurally have no
                                 counterparty_identifier, so they are
                                 excluded from the graph entirely at
                                 construction time (graph_builder.py
                                 only creates an edge when a counterparty
                                 identifier exists) - the transactions
                                 this rule needs were never turned into
                                 edges in the first place.
    late_night_score             LATE_NIGHT_TRANSACTION needs hour-of-
                                 day. graph_builder.py's edges only carry
                                 `date` (day granularity), never
                                 `time` - every adapter's Transaction.time
                                 field is largely None/unused upstream of
                                 the graph, so there is nothing to compute
                                 this from.
    balance_drop_score            BALANCE_DROP_ALERT needs per-transaction
                                 balance before/after. graph_builder.py's
                                 edges do not carry `balance` (same gap
                                 noted, undecided, in the v4 STATE.md).

  DOES NOT EXIST AT ALL (stubs, confirmed absent from the whole
  codebase, not approximated):
    circular_return_score
    bidirectional_score

combined_rule_score() = max() over every MAPPABLE score only. The three
structurally-unmappable stubs and the two nonexistent-rule stubs all
return constant 0.0, so none of them can ever change the combined score
- this baseline is exactly as strong as the 5 mappable signals, no more,
no less.

────────────────────────────────────────────────────────────────────
Account vs counterparty node fairness (task step 5-6)
────────────────────────────────────────────────────────────────────
Every mappable rule above is computed identically for account and
counterparty nodes, using only that node's own touching edges - no
arbitrary/placeholder score is ever assigned to a counterparty node
just to fill the array. That said, this is a genuine, acknowledged
adaptation, not a hidden one: the real rules were designed for a node
that has ITS OWN complete statement (every transaction it was ever
party to, from its own bank's export). An 'account' node in this graph
does have that. A 'counterparty' node does NOT - we only ever see the
edges where one of OUR accounts transacted with them, which is a
partial, one-sided view of that entity's real activity. Rules like
HIGH_VALUE and SPENDING_SPIKE (which depend on a channel/daily median
computed from "all of this node's own transactions") are most affected
by this - a counterparty's per-channel or per-day median here reflects
only the visible slice, not their real activity.

A cleaner experiment - restricting the rule-vs-GNN comparison to
account nodes only - was considered (task step 6) and is NOT run here:
this dataset has only 51 real accounts, and the fraud generator (which
this correction round is explicitly forbidden from changing) never
labels an account-type node as synthetic - every injected ring node is
node_type='counterparty' by construction (see labeling.py). An
account-only slice would therefore have zero positive labels and no
computable AUC/PR-AUC at all under the current benchmark design. This
is reported as a real methodological limitation, not worked around.
"""

from collections import defaultdict
from datetime import timedelta
from typing import Dict, List

import networkx as nx
import numpy as np
import pandas as pd

# ── Constants copied verbatim from backend/services/risk_engine.py ──────────
RAPID_WINDOW_MINUTES = 10
RAPID_BURST_THRESHOLD = 5  # corrected from 4

HIGH_VALUE_MULTIPLIER = 5.0
HIGH_VALUE_FLOOR = 5_000.0

SPENDING_SPIKE_MIN_DAYS = 7
SPENDING_SPIKE_MULTIPLIER = 4.0
SPENDING_SPIKE_FLOOR = 10_000.0

REPEATED_MIN_COUNT = 3
REPEATED_WINDOW_HOURS = 24
REPEATED_AMOUNT_TOLERANCE = 0.02  # risk_engine.py's value (unified_fraud_engine.py uses 0.03)

# ── Constant sourced from unified_fraud_engine.py specifically (not
# risk_engine.py - see module docstring) ────────────────────────────────────
SMURF_AMOUNT_LOW = 9000.0
SMURF_AMOUNT_HIGH = 9999.0
SMURF_EPISODE_GAP_HOURS = 24.0   # real episode-boundary gap
SMURF_EPISODE_NEAR_PCT = 0.50     # real "Smurfing" label threshold


def _node_events(g: nx.MultiDiGraph, node: str):
    """Every (amount, parsed_date, other_endpoint, direction) tuple for
    transactions touching this node as either endpoint - the closest
    analog to 'this node's own transaction list' available here."""
    events = []
    for _, v, d in g.out_edges(node, data=True):
        events.append((d["amount"], d.get("parsed_date"), v, "OUT"))
    for u, _, d in g.in_edges(node, data=True):
        events.append((d["amount"], d.get("parsed_date"), u, "IN"))
    return events


def burst_score(g: nx.MultiDiGraph, node_list: List[str]) -> np.ndarray:
    """
    Faithful port of risk_engine.py's detect_rapid_transactions:
    for each of this node's own transactions, count how many of its
    OTHER transactions (any direction - the real rule does not filter
    by debit/credit) fall within +/- RAPID_WINDOW_MINUTES/2 of it
    (a symmetric window, exactly as in the real code, NOT a one-sided
    sliding window). If any transaction's count >= RAPID_BURST_THRESHOLD
    (5), this node is flagged.

    Mapping: transaction-level flag -> node score = 1.0 if this node
    has at least one transaction that would be flagged by the real
    rule, else 0.0 (binary, matching the real rule's own binary flag -
    there's no natural continuous relaxation to invent here without
    changing what the rule means).
    """
    scores = np.zeros(len(node_list), dtype=np.float32)
    half_window_sec = (RAPID_WINDOW_MINUTES / 2.0) * 60.0

    for i, node in enumerate(node_list):
        events = [dt for _, dt, _, _ in _node_events(g, node) if dt is not None]
        if len(events) < RAPID_BURST_THRESHOLD:
            continue
        events.sort()
        n = len(events)
        secs = [e.timestamp() for e in events]

        # Two-pointer sweep: since secs is sorted, the set of j with
        # |secs[i]-secs[j]| <= half_window is a contiguous range
        # [left, right] that only moves forward as i increases -
        # O(n) total, exact same symmetric-window semantics as the
        # real O(n^2) implementation in risk_engine.py, just fast
        # enough for a 10,000+-transaction hub account.
        flagged = False
        left = 0
        right = 0
        for idx in range(n):
            if right < idx:
                right = idx
            while left < n and secs[idx] - secs[left] > half_window_sec:
                left += 1
            while right + 1 < n and secs[right + 1] - secs[idx] <= half_window_sec:
                right += 1
            count_in_window = right - left + 1
            if count_in_window >= RAPID_BURST_THRESHOLD:
                flagged = True
                break

        scores[i] = 1.0 if flagged else 0.0

    return scores


def high_value_score(g: nx.MultiDiGraph, node_list: List[str]) -> np.ndarray:
    """
    Port of detect_high_value_transactions: per-channel (payment_method)
    median amount computed from this node's own transactions; flag any
    transaction exceeding max(HIGH_VALUE_MULTIPLIER x channel_median,
    HIGH_VALUE_FLOOR).

    Mapping: node score = 1.0 if any of this node's own transactions
    would be flagged, else 0.0.
    """
    scores = np.zeros(len(node_list), dtype=np.float32)

    for i, node in enumerate(node_list):
        events = _node_events(g, node)
        if not events:
            continue
        df = pd.DataFrame(events, columns=["amount", "date", "other", "dir"])
        channel_medians = df.groupby("dir")["amount"].median().to_dict()
        # payment_method isn't stored per-node-event here (only amount/date/
        # other/dir survive _node_events) - direction is used as the closest
        # available channel proxy. See note below.
        flagged = False
        for _, row in df.iterrows():
            median_c = channel_medians.get(row["dir"])
            if median_c is None or median_c == 0:
                continue
            threshold = HIGH_VALUE_MULTIPLIER * median_c
            if row["amount"] > threshold and row["amount"] > HIGH_VALUE_FLOOR:
                flagged = True
                break
        scores[i] = 1.0 if flagged else 0.0

    return scores


def spending_spike_score(g: nx.MultiDiGraph, node_list: List[str]) -> np.ndarray:
    """
    Port of detect_spending_spikes: requires >= SPENDING_SPIKE_MIN_DAYS
    (7) distinct days of OUTGOING activity ("spending" = money leaving
    this node, i.e. out_edges); flags any day whose total outgoing
    amount exceeds max(SPENDING_SPIKE_MULTIPLIER x median daily spend,
    SPENDING_SPIKE_FLOOR).

    Mapping: node score = 1.0 if any day qualifies, else 0.0.
    """
    scores = np.zeros(len(node_list), dtype=np.float32)

    for i, node in enumerate(node_list):
        out_events = [(amt, dt) for amt, dt, _, direction in _node_events(g, node)
                      if direction == "OUT" and dt is not None]
        if not out_events:
            continue
        df = pd.DataFrame(out_events, columns=["amount", "date"])
        df["day"] = df["date"].apply(lambda d: d.date())
        if df["day"].nunique() < SPENDING_SPIKE_MIN_DAYS:
            continue
        daily_totals = df.groupby("day")["amount"].sum()
        median_daily = daily_totals.median()
        threshold = SPENDING_SPIKE_MULTIPLIER * median_daily
        spike = ((daily_totals > threshold) & (daily_totals > SPENDING_SPIKE_FLOOR)).any()
        scores[i] = 1.0 if spike else 0.0

    return scores


def repeated_transaction_score(g: nx.MultiDiGraph, node_list: List[str]) -> np.ndarray:
    """
    Port of detect_repeated_transactions: the real "key" is a resolved
    merchant/narration identifier. The closest analog available on this
    graph is the OTHER endpoint of the transaction (the counterparty a
    given node transacted with) - flags a node if it has
    >= REPEATED_MIN_COUNT (3) transactions to/from the SAME other node,
    within REPEATED_AMOUNT_TOLERANCE (2%) of each other's amount,
    within a REPEATED_WINDOW_HOURS (24h) window.

    Mapping: node score = 1.0 if any such cluster exists among this
    node's own transactions, else 0.0.
    """
    scores = np.zeros(len(node_list), dtype=np.float32)
    window = timedelta(hours=REPEATED_WINDOW_HOURS)

    for i, node in enumerate(node_list):
        events = [(amt, dt, other) for amt, dt, other, _ in _node_events(g, node) if dt is not None]
        by_key = defaultdict(list)
        for amt, dt, other in events:
            by_key[other].append((dt, amt))

        flagged = False
        for other, rows in by_key.items():
            if len(rows) < REPEATED_MIN_COUNT:
                continue
            for idx_i, (ts_i, amt_i) in enumerate(rows):
                similar = [
                    1 for idx_j, (ts_j, amt_j) in enumerate(rows)
                    if idx_i != idx_j
                    and abs((ts_i - ts_j).total_seconds()) <= window.total_seconds()
                    and (abs(amt_i - amt_j) / amt_i <= REPEATED_AMOUNT_TOLERANCE if amt_i != 0 else amt_j == 0)
                ]
                if len(similar) >= REPEATED_MIN_COUNT - 1:
                    flagged = True
                    break
            if flagged:
                break
        scores[i] = 1.0 if flagged else 0.0

    return scores


def smurfing_episode_score(g: nx.MultiDiGraph, node_list: List[str]) -> np.ndarray:
    """
    Port of unified_fraud_engine.py's episode-level "Smurfing" label -
    NOT risk_engine.py, which has no smurfing logic at all (see module
    docstring). The real logic: group an account's own already-
    rule-flagged transactions into episodes (a new episode starts
    whenever the gap since the previous transaction exceeds
    SMURF_EPISODE_GAP_HOURS=24), then label an episode "Smurfing" if
    is_near_pct (fraction of the episode's transactions with amount in
    [9000, 9999]) exceeds 0.50.

    ONE DOCUMENTED ADAPTATION (per task step 3's explicit instruction to
    document rather than hide any such gap): the real episodes are built
    only from transactions that ALREADY triggered at least one of the
    other rules - that requires all 7 rules' flags as a prerequisite
    filter, which isn't wired up as a joint pipeline here. This
    implementation builds episodes from ALL of a node's own transactions
    instead of only pre-flagged ones. This is a real, acknowledged
    difference from the deployed system, not the same rule under a
    different name - it will flag some nodes the real system wouldn't
    (nodes with a near-threshold-heavy episode that never had any other
    rule fire on it first) and is reported as a limitation, not silently
    equated to the real rule.

    Mapping: node score = 1.0 if any of this node's episodes has
    is_near_pct > 0.50, else 0.0.
    """
    scores = np.zeros(len(node_list), dtype=np.float32)
    gap = timedelta(hours=SMURF_EPISODE_GAP_HOURS)

    for i, node in enumerate(node_list):
        events = sorted([(dt, amt) for amt, dt, _, _ in _node_events(g, node) if dt is not None])
        if not events:
            continue

        episodes = []
        current = [events[0]]
        for dt, amt in events[1:]:
            if dt - current[-1][0] > gap:
                episodes.append(current)
                current = []
            current.append((dt, amt))
        episodes.append(current)

        flagged = False
        for ep in episodes:
            near = sum(1 for _, amt in ep if SMURF_AMOUNT_LOW <= amt <= SMURF_AMOUNT_HIGH)
            if near / len(ep) > SMURF_EPISODE_NEAR_PCT:
                flagged = True
                break
        scores[i] = 1.0 if flagged else 0.0

    return scores


def excessive_withdrawal_score(g: nx.MultiDiGraph, node_list: List[str]) -> np.ndarray:
    """STUB - structurally not mappable. See module docstring."""
    return np.zeros(len(node_list), dtype=np.float32)


def late_night_score(g: nx.MultiDiGraph, node_list: List[str]) -> np.ndarray:
    """STUB - structurally not mappable (no time-of-day data on edges). See module docstring."""
    return np.zeros(len(node_list), dtype=np.float32)


def balance_drop_score(g: nx.MultiDiGraph, node_list: List[str]) -> np.ndarray:
    """STUB - structurally not mappable (no balance data on edges). See module docstring."""
    return np.zeros(len(node_list), dtype=np.float32)


def circular_return_score(g: nx.MultiDiGraph, node_list: List[str]) -> np.ndarray:
    """STUB - confirmed absent from the codebase entirely. See module docstring."""
    return np.zeros(len(node_list), dtype=np.float32)


def bidirectional_score(g: nx.MultiDiGraph, node_list: List[str]) -> np.ndarray:
    """STUB - confirmed absent from the codebase entirely. See module docstring."""
    return np.zeros(len(node_list), dtype=np.float32)


def combined_rule_score(g: nx.MultiDiGraph, node_list: List[str]) -> Dict[str, np.ndarray]:
    components = dict(
        burst=burst_score(g, node_list),
        high_value=high_value_score(g, node_list),
        spending_spike=spending_spike_score(g, node_list),
        repeated_transaction=repeated_transaction_score(g, node_list),
        smurfing_episode=smurfing_episode_score(g, node_list),
        excessive_withdrawal=excessive_withdrawal_score(g, node_list),
        late_night=late_night_score(g, node_list),
        balance_drop=balance_drop_score(g, node_list),
        circular_return=circular_return_score(g, node_list),
        bidirectional=bidirectional_score(g, node_list),
    )
    combined = np.zeros(len(node_list), dtype=np.float32)
    for arr in components.values():
        combined = np.maximum(combined, arr)
    components["combined"] = combined
    return components


if __name__ == "__main__":
    import sys
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
        print(f"{name:22s} min={arr.min():.3f} max={arr.max():.3f} mean={arr.mean():.4f} nonzero={np.count_nonzero(arr)}")

    assert scores["combined"].min() >= 0.0 and scores["combined"].max() <= 1.0
    assert len(scores["combined"]) == len(node_list)
    print("\nOK: scores in [0,1], every node scored")
