"""
Synthetic fraud-ring injection - Label Strategy C from STATE.md.

Since there is no ground-truth fraud label in this dataset, injected
synthetic rings are the only way to sanity-check whether the embeddings
/ anomaly scores carry any signal at all before trusting them on real
(unlabeled) nodes. This is a validation harness, not a fraud label for
real nodes - see README "What 'fraud' means in this pipeline".

Three typologies, each built from brand-new synthetic nodes (so ground
truth is unambiguous) and bridged with 1-2 edges into randomly chosen
*real* nodes (so the injected structure isn't a trivially-isolated
component the model could "detect" by connectivity alone):

  cycle     A -> B -> C -> (-> D) -> A, near-equal amounts, tight window
  fan_out   one hub -> K new leaves, near-equal amounts, tight window
            (structuring / smurfing)
  fan_in    K new leaves -> one hub, then hub -> real node (mule
            aggregation + cash-out)

Every synthetic node/edge is tagged is_synthetic=True so it can be
stripped back out or evaluated against separately at any point.
"""

import random
from datetime import datetime, timedelta
from typing import List, Tuple

import networkx as nx

SYNTHETIC_PREFIX = "SYN:"


def _synth_node(ring_id: int, idx: int) -> str:
    return f"{SYNTHETIC_PREFIX}R{ring_id}_N{idx}"


def _pick_base_date(g: nx.MultiDiGraph, rng: random.Random) -> datetime:
    dates = [
        d["parsed_date"] for _, _, d in g.edges(data=True)
        if d.get("parsed_date") is not None
    ]
    if not dates:
        return datetime(2025, 6, 1)
    return rng.choice(dates)


def _pick_real_nodes(g: nx.MultiDiGraph, rng: random.Random, k: int) -> List[str]:
    real_nodes = [n for n, d in g.nodes(data=True) if not d.get("is_synthetic")]
    if not real_nodes:
        return []
    return rng.sample(real_nodes, k=min(k, len(real_nodes)))


def _add_synthetic_edge(g, u, v, amount, date, ring_id, ring_type):
    if not g.has_node(u):
        g.add_node(u, node_type="counterparty", label=f"synthetic:{u}", is_synthetic=True,
                   ring_id=ring_id, ring_type=ring_type)
    if not g.has_node(v):
        g.add_node(v, node_type="counterparty", label=f"synthetic:{v}", is_synthetic=True,
                   ring_id=ring_id, ring_type=ring_type)

    g.add_edge(
        u, v,
        amount=float(amount),
        date=date.strftime("%d-%m-%Y"),
        parsed_date=date,
        payment_method="UPI",
        transaction_id=f"SYNTH:{ring_id}:{u}:{v}:{date.isoformat()}",
        is_reversal=False,
        source_file="__synthetic__",
        is_synthetic=True,
        ring_id=ring_id,
        ring_type=ring_type,
    )


def _inject_cycle(g, ring_id, rng, size=4):
    base_amount = rng.uniform(8000, 45000)
    base_date = _pick_base_date(g, rng)
    nodes = [_synth_node(ring_id, i) for i in range(size)]

    amount = base_amount
    for i in range(size):
        u, v = nodes[i], nodes[(i + 1) % size]
        date = base_date + timedelta(hours=rng.uniform(0, 48) * i)
        _add_synthetic_edge(g, u, v, amount, date, ring_id, "cycle")
        amount *= rng.uniform(0.95, 1.02)  # small skim/fee each hop

    bridge_targets = _pick_real_nodes(g, rng, 1)
    for target in bridge_targets:
        _add_synthetic_edge(g, nodes[0], target, base_amount * 0.3,
                             base_date + timedelta(days=1), ring_id, "cycle")

    return nodes


def _inject_fan_out(g, ring_id, rng, leaves=5):
    hub = _synth_node(ring_id, 0)
    base_amount = rng.uniform(9000, 48000)
    base_date = _pick_base_date(g, rng)
    nodes = [hub]

    real_source = _pick_real_nodes(g, rng, 1)
    if real_source:
        _add_synthetic_edge(g, real_source[0], hub, base_amount * leaves * 0.9,
                             base_date, ring_id, "fan_out")

    for i in range(1, leaves + 1):
        leaf = _synth_node(ring_id, i)
        nodes.append(leaf)
        amount = base_amount * rng.uniform(0.97, 1.03)
        date = base_date + timedelta(hours=rng.uniform(0, 30))
        _add_synthetic_edge(g, hub, leaf, amount, date, ring_id, "fan_out")

    return nodes


def _inject_fan_in(g, ring_id, rng, leaves=5):
    hub = _synth_node(ring_id, 0)
    base_amount = rng.uniform(7000, 40000)
    base_date = _pick_base_date(g, rng)
    nodes = [hub]

    for i in range(1, leaves + 1):
        leaf = _synth_node(ring_id, i)
        nodes.append(leaf)
        amount = base_amount * rng.uniform(0.95, 1.05)
        date = base_date + timedelta(hours=rng.uniform(0, 24))
        _add_synthetic_edge(g, leaf, hub, amount, date, ring_id, "fan_in")

    cash_out_targets = _pick_real_nodes(g, rng, 1)
    if cash_out_targets:
        _add_synthetic_edge(g, hub, cash_out_targets[0], base_amount * leaves * 0.92,
                             base_date + timedelta(hours=36), ring_id, "fan_in")

    return nodes


INJECTORS = {
    "cycle": _inject_cycle,
    "fan_out": _inject_fan_out,
    "fan_in": _inject_fan_in,
}


def inject_synthetic_rings(
    g: nx.MultiDiGraph,
    n_rings: int = 12,
    seed: int = 13,
) -> Tuple[nx.MultiDiGraph, List[dict]]:
    """
    Mutates g in place (adds synthetic nodes/edges only - never removes
    or alters a real transaction) and returns a record of every
    injected ring for later evaluation.
    """

    rng = random.Random(seed)
    ring_types = list(INJECTORS.keys())
    records = []

    for ring_id in range(n_rings):
        ring_type = ring_types[ring_id % len(ring_types)]
        size = rng.randint(4, 6)
        nodes = INJECTORS[ring_type](g, ring_id, rng, size)
        records.append(dict(ring_id=ring_id, ring_type=ring_type, nodes=nodes))

    return g, records


if __name__ == "__main__":
    import pickle
    import sys
    sys.path.insert(0, ".")
    from graph_builder import build_multigraph

    with open("_txns_cache.pkl", "rb") as f:
        txns = pickle.load(f)

    g, report = build_multigraph(txns)
    n_before, e_before = g.number_of_nodes(), g.number_of_edges()

    g, records = inject_synthetic_rings(g, n_rings=12)

    n_after, e_after = g.number_of_nodes(), g.number_of_edges()
    print(f"Nodes: {n_before} -> {n_after} (+{n_after - n_before})")
    print(f"Edges: {e_before} -> {e_after} (+{e_after - e_before})")
    for r in records:
        print(f"  ring {r['ring_id']:2d} [{r['ring_type']:8s}] {len(r['nodes'])} nodes")
