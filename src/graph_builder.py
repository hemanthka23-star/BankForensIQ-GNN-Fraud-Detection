"""
Build the transaction graph from a list of Transaction objects.

Node types
----------
account       one per bank account_id seen in the ingested statements
counterparty  one per distinct counterparty_identifier (UPI handle, NEFT
              sender code, IMPS name, ...) found by counterparty_extractor

A transaction only becomes a graph edge when the extractor already
produced a counterparty_identifier for it - transactions with no
identifiable counterparty (ATM withdrawals, interest credit, charges,
and anything the extractor couldn't parse - e.g. most RTGS right now)
are counted and reported, but never turned into a fabricated node/edge.

Two views are built:
  - `multigraph` : networkx.MultiDiGraph, one edge per transaction,
                    full attributes kept - used for feature engineering
                    and for rendering subgraphs.
  - `adjacency`   : simple, symmetric, weighted N x N numpy adjacency
                    (weight = number of transactions between the pair),
                    aligned to `node_list` - used by the GNN.
"""

import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import networkx as nx
import numpy as np

from transaction_schema import Transaction

ACCOUNT_PREFIX = "ACC:"
COUNTERPARTY_PREFIX = "CP:"


def _account_node(account_id: str) -> str:
    return f"{ACCOUNT_PREFIX}{account_id}"


def _counterparty_node(identifier: str) -> str:
    return f"{COUNTERPARTY_PREFIX}{identifier.strip().upper()}"


def _parse_date(date_str: Optional[str]):
    if not date_str:
        return None
    date_str = date_str.strip()
    for fmt in ("%d-%m-%Y", "%Y-%m-%d", "%d/%m/%Y", "%d-%b-%Y", "%d-%b-%y"):
        try:
            return datetime.strptime(date_str[:11].strip(), fmt)
        except ValueError:
            continue
    return None


@dataclass
class GraphBuildReport:
    total_transactions: int = 0
    edges_built: int = 0
    skipped_no_identifier: int = 0
    skipped_no_amount: int = 0
    skipped_bad_date: int = 0

    @property
    def summary(self) -> str:
        return (
            f"Total transactions considered : {self.total_transactions}\n"
            f"Edges built (has counterparty) : {self.edges_built}\n"
            f"Skipped - no counterparty id   : {self.skipped_no_identifier}\n"
            f"Skipped - no usable amount     : {self.skipped_no_amount}\n"
            f"Skipped - unparseable date     : {self.skipped_bad_date}"
        )


def build_multigraph(transactions: List[Transaction]) -> Tuple[nx.MultiDiGraph, GraphBuildReport]:
    g = nx.MultiDiGraph()
    report = GraphBuildReport(total_transactions=len(transactions))

    for t in transactions:
        if not t.counterparty_identifier:
            report.skipped_no_identifier += 1
            continue

        if t.amount is None:
            report.skipped_no_amount += 1
            continue

        acc_node = _account_node(t.account_id)
        cp_node = _counterparty_node(t.counterparty_identifier)

        if not g.has_node(acc_node):
            g.add_node(acc_node, node_type="account", label=t.account_id, is_synthetic=False)

        if not g.has_node(cp_node):
            g.add_node(
                cp_node,
                node_type="counterparty",
                label=t.counterparty_name or t.counterparty_label or t.counterparty_identifier,
                bank=t.counterparty_bank,
                is_synthetic=False,
            )

        parsed_date = _parse_date(t.date)
        if parsed_date is None:
            report.skipped_bad_date += 1

        edge_attrs = dict(
            amount=float(t.amount),
            date=t.date,
            parsed_date=parsed_date,
            payment_method=t.payment_method,
            transaction_id=t.transaction_id,
            is_reversal=bool(t.is_reversal),
            source_file=t.source_file,
            is_synthetic=False,
        )

        if t.direction == "DEBIT":
            g.add_edge(acc_node, cp_node, **edge_attrs)
        else:
            g.add_edge(cp_node, acc_node, **edge_attrs)

        report.edges_built += 1

    return g, report


def build_adjacency(g: nx.MultiDiGraph) -> Tuple[np.ndarray, List[str]]:
    """
    Collapse the MultiDiGraph into a simple, symmetric, weighted
    adjacency matrix for GNN message passing. Weight = number of
    transactions between the pair (direction collapsed - the GNN
    treats "moved money together" as the relation; direction and
    amount are kept as edge/node *features* instead, see features.py).
    """

    node_list = list(g.nodes())
    index = {n: i for i, n in enumerate(node_list)}
    n = len(node_list)

    adj = np.zeros((n, n), dtype=np.float32)

    for u, v in g.edges():
        i, j = index[u], index[v]
        adj[i, j] += 1.0
        adj[j, i] += 1.0

    return adj, node_list


def build_directional_channels(g: nx.MultiDiGraph, node_list: List[str] = None):
    """
    4 row-normalized weighted DIRECTED adjacency matrices, for a
    direction- and edge-weight-aware GNN (see gnn_model.py). Each row
    i sums to 1 over i's neighbors in that channel, i.e. this is
    GraphSAGE-mean-style aggregation, split by direction and by two
    edge-weightings:

        out_amt[i, j]   i's share of node i's total OUTGOING amount
                        that went to j (money i sent, weighted by size)
        out_cnt[i, j]   i's share of node i's OUTGOING transaction
                        count that went to j (weighted by frequency)
        in_amt, in_cnt  the symmetric pair for INCOMING edges

    This replaces the single binary symmetric adjacency for the parts
    of the model that need to know not just "connected" but "who sent
    how much, how often, in which direction" - directly addresses
    "stop throwing away transaction direction" / "add edge features"
    from the review.
    """
    node_list = node_list or list(g.nodes())
    index = {n: i for i, n in enumerate(node_list)}
    n = len(node_list)

    amt = np.zeros((n, n), dtype=np.float64)
    cnt = np.zeros((n, n), dtype=np.float64)

    for u, v, d in g.edges(data=True):
        i, j = index[u], index[v]
        amt[i, j] += np.log1p(d.get("amount", 0.0))
        cnt[i, j] += 1.0

    def _row_normalize(m):
        row_sum = m.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        return (m / row_sum).astype(np.float32)

    out_amt = _row_normalize(amt)
    out_cnt = _row_normalize(cnt)
    in_amt = _row_normalize(amt.T)
    in_cnt = _row_normalize(cnt.T)

    return dict(out_amt=out_amt, out_cnt=out_cnt, in_amt=in_amt, in_cnt=in_cnt)


def directed_edge_index(g: nx.MultiDiGraph, node_list: List[str]):
    """Deduped (i, j) index arrays for every distinct directed pair
    that has at least one real transaction - the positive set for the
    direction-aware decoder's training loss."""
    index = {n: i for i, n in enumerate(node_list)}
    pairs = set(g.edges())
    if not pairs:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    i_idx = np.array([index[u] for u, v in pairs], dtype=np.int64)
    j_idx = np.array([index[v] for u, v in pairs], dtype=np.int64)
    return i_idx, j_idx


if __name__ == "__main__":
    import pickle
    import sys

    with open(sys.argv[1] if len(sys.argv) > 1 else "_txns_cache.pkl", "rb") as f:
        txns = pickle.load(f)

    g, report = build_multigraph(txns)
    print(report.summary)
    print(f"\nNodes: {g.number_of_nodes()}  Edges: {g.number_of_edges()}")

    adj, nodes = build_adjacency(g)
    print(f"Adjacency shape: {adj.shape}, nonzero entries: {np.count_nonzero(adj)}")
