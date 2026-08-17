"""
Synthetic fraud-ring injection v2 - systematic, harder-to-cheat benchmark.

v1 (kept as labeling_v1_backup.py) generated 3 typologies with tight
timing and near-identical amounts. External review of this project
called that out directly: "of course the GNN detects artificially
injected, highly structured rings." This version is the direct response.

5 typologies, each parametrized rather than hardcoded:

  cycle             A -> B -> C -> ... -> A
  fan_out           hub -> many leaves               ("structuring")
  fan_in            many leaves -> hub -> real sink   ("mule aggregation")
  probe_and_drain   tiny probe txn, gap, then a large drain txn, then
                    rapid cash-out
  mule_chain        linear (non-cyclic) chain, amount decaying per hop,
                    spread over days                  ("layering")

3 difficulty tiers (easy / medium / hard) controlling:
  - amount_variation   how far each hop's amount can drift (5% -> 50%)
  - timing_spread      how loosely spread the transactions are in time
                        (hours -> ~1.5 weeks)
  - noise_prob/edges   chance + count of extra edges from ring nodes to
                        random *real* nodes, with amounts drawn from the
                        real amount distribution - "fraud mixed with
                        normal transactions", so rings aren't only ever
                        connected via obviously-patterned edges.
  - size_range         easy tiers use smaller rings, hard tiers larger
                        ones (3-5 up to 8-15 nodes)

Partial ("core") ground truth: not every node in a ring is an equally
strong fraud signal. `mule_chain` and `probe_and_drain` mark only the
middle pass-through node(s) as `is_core` - matching "fraud where only
1-2 nodes are unusual" from the review. Evaluation can report recovery
of *any ring member* vs *core members only* separately.

Still 100% additive - a real transaction is never modified or removed.

v3 note: `size_range` is now constant across all three tiers (see TIERS
below) - v2 let it grow with difficulty, which confounded "harder" with
"structurally bigger". `probe_and_drain` is always exactly 2 nodes
regardless of tier (it's a fixed 3-hop topology, not a size-parametrized
one), so it was never affected by that confound either way.
"""

import random
from datetime import datetime, timedelta
from typing import List, Tuple, Dict

import networkx as nx

SYNTHETIC_PREFIX = "SYN:"

    # v2 had size_range grow with difficulty (3-5 -> 5-9 -> 8-15), which
    # confounded "harder" with "bigger" - STATE.md's own v2 finding was
    # that hard-tier nodes scored as MORE detectable than easy-tier ones,
    # backwards from intent, because bigger rings simply have more raw
    # degree, independent of whether amounts/timing are actually subtler.
    # Fixed here (v3) by holding size_range constant across all three
    # tiers, so only amount_variation / timing_spread / noise vary -
    # "difficulty" now isolates subtlety specifically, not structure size.
TIERS = {
    "easy": dict(
        amount_variation=0.05, timing_spread_hours=3,
        noise_prob=0.0, noise_edges=(0, 0), size_range=(6, 8),
    ),
    "medium": dict(
        amount_variation=0.20, timing_spread_hours=72,
        noise_prob=0.5, noise_edges=(1, 2), size_range=(6, 8),
    ),
    "hard": dict(
        amount_variation=0.50, timing_spread_hours=24 * 10,
        noise_prob=0.85, noise_edges=(2, 4), size_range=(6, 8),
    ),
}

# How this maps onto the terms the review used, since we parametrize
# instead of hardcoding a function per named pattern:
#   "structuring"           -> fan_out,  medium/hard tier
#   "delayed circular"       -> cycle,    medium/hard tier
#   "coordinated burst"       -> fan_in,   easy/medium tier (tight timing)
#   "layering"                -> mule_chain, any tier
#   "small->large->cashout"    -> probe_and_drain

TYPES = ["cycle", "fan_out", "fan_in", "probe_and_drain", "mule_chain"]


def _synth_node(ring_id: int, idx: int) -> str:
    return f"{SYNTHETIC_PREFIX}R{ring_id}_N{idx}"


def _jitter_amount(base: float, variation: float, rng: random.Random) -> float:
    return max(base * (1.0 + rng.uniform(-variation, variation)), 1.0)


def _jitter_time(base: datetime, spread_hours: float, rng: random.Random) -> datetime:
    return base + timedelta(hours=rng.uniform(0, spread_hours))


def _pick_base_date(g: nx.MultiDiGraph, rng: random.Random) -> datetime:
    dates = [d["parsed_date"] for _, _, d in g.edges(data=True) if d.get("parsed_date") is not None]
    return rng.choice(dates) if dates else datetime(2025, 6, 1)


def _pick_real_nodes(g: nx.MultiDiGraph, rng: random.Random, k: int) -> List[str]:
    real_nodes = [n for n, d in g.nodes(data=True) if not d.get("is_synthetic")]
    return rng.sample(real_nodes, k=min(k, len(real_nodes))) if real_nodes else []


def _add_synthetic_edge(g, u, v, amount, date, ring_id, ring_type, is_noise=False):
    for node in (u, v):
        if not g.has_node(node):
            g.add_node(node, node_type="counterparty", label=f"synthetic:{node}",
                       is_synthetic=True, ring_id=ring_id, ring_type=ring_type)

    g.add_edge(
        u, v,
        amount=float(amount),
        date=date.strftime("%d-%m-%Y"),
        parsed_date=date,
        payment_method="UPI",
        transaction_id=f"SYNTH:{ring_id}:{u}:{v}:{date.isoformat()}:{is_noise}",
        is_reversal=False,
        source_file="__synthetic__",
        is_synthetic=True,
        ring_id=ring_id,
        ring_type=ring_type,
        is_noise=is_noise,
    )


def _add_background_noise(g, ring_nodes, tier, base_date, ring_id, ring_type,
                           real_amount_sample, rng: random.Random):
    """
    Attach a few edges from ring nodes to random *real* nodes, with
    amounts drawn from the real amount distribution (not the fraud
    pattern), scattered widely in time - "fraud mixed with normal
    transactions". Skipped entirely at the easy tier.
    """
    if tier["noise_prob"] <= 0 or not real_amount_sample:
        return

    lo, hi = tier["noise_edges"]
    for node in ring_nodes:
        if rng.random() > tier["noise_prob"]:
            continue
        n_edges = rng.randint(lo, hi)
        for _ in range(n_edges):
            real_partner = _pick_real_nodes(g, rng, 1)
            if not real_partner:
                continue
            amount = rng.choice(real_amount_sample)
            date = _jitter_time(base_date, tier["timing_spread_hours"] * 3, rng)
            if rng.random() < 0.5:
                _add_synthetic_edge(g, node, real_partner[0], amount, date, ring_id, ring_type, is_noise=True)
            else:
                _add_synthetic_edge(g, real_partner[0], node, amount, date, ring_id, ring_type, is_noise=True)


def _inject_cycle(g, ring_id, rng, tier, real_amount_sample):
    size = rng.randint(*tier["size_range"])
    base_amount = rng.uniform(8000, 45000)
    base_date = _pick_base_date(g, rng)
    nodes = [_synth_node(ring_id, i) for i in range(size)]

    amount = base_amount
    for i in range(size):
        u, v = nodes[i], nodes[(i + 1) % size]
        date = _jitter_time(base_date, tier["timing_spread_hours"], rng) + timedelta(hours=6 * i)
        _add_synthetic_edge(g, u, v, amount, date, ring_id, "cycle")
        amount = _jitter_amount(amount, tier["amount_variation"], rng)

    for target in _pick_real_nodes(g, rng, 1):
        _add_synthetic_edge(g, nodes[0], target, base_amount * 0.3,
                             base_date + timedelta(days=1), ring_id, "cycle")

    _add_background_noise(g, nodes, tier, base_date, ring_id, "cycle", real_amount_sample, rng)
    return nodes, list(nodes)  # all nodes are core


def _inject_fan_out(g, ring_id, rng, tier, real_amount_sample):
    leaves = rng.randint(*tier["size_range"])
    hub = _synth_node(ring_id, 0)
    base_amount = rng.uniform(9000, 48000)
    base_date = _pick_base_date(g, rng)
    nodes = [hub]

    for target in _pick_real_nodes(g, rng, 1):
        _add_synthetic_edge(g, target, hub, base_amount * leaves * 0.9, base_date, ring_id, "fan_out")

    for i in range(1, leaves + 1):
        leaf = _synth_node(ring_id, i)
        nodes.append(leaf)
        amount = _jitter_amount(base_amount, tier["amount_variation"], rng)
        date = _jitter_time(base_date, tier["timing_spread_hours"], rng)
        _add_synthetic_edge(g, hub, leaf, amount, date, ring_id, "fan_out")

    _add_background_noise(g, nodes, tier, base_date, ring_id, "fan_out", real_amount_sample, rng)
    return nodes, list(nodes)


def _inject_fan_in(g, ring_id, rng, tier, real_amount_sample):
    leaves = rng.randint(*tier["size_range"])
    hub = _synth_node(ring_id, 0)
    base_amount = rng.uniform(7000, 40000)
    base_date = _pick_base_date(g, rng)
    nodes = [hub]

    for i in range(1, leaves + 1):
        leaf = _synth_node(ring_id, i)
        nodes.append(leaf)
        amount = _jitter_amount(base_amount, tier["amount_variation"], rng)
        date = _jitter_time(base_date, tier["timing_spread_hours"], rng)
        _add_synthetic_edge(g, leaf, hub, amount, date, ring_id, "fan_in")

    for target in _pick_real_nodes(g, rng, 1):
        _add_synthetic_edge(g, hub, target, base_amount * leaves * 0.92,
                             base_date + timedelta(hours=tier["timing_spread_hours"] + 6),
                             ring_id, "fan_in")

    _add_background_noise(g, nodes, tier, base_date, ring_id, "fan_in", real_amount_sample, rng)
    return nodes, list(nodes)


def _inject_probe_and_drain(g, ring_id, rng, tier, real_amount_sample):
    node = _synth_node(ring_id, 0)  # the drop/pass-through account
    prober = _synth_node(ring_id, 1)
    base_date = _pick_base_date(g, rng)
    nodes = [node, prober]

    probe_amount = rng.uniform(10, 500)
    _add_synthetic_edge(g, prober, node, probe_amount, base_date, ring_id, "probe_and_drain")

    gap = max(tier["timing_spread_hours"], 6)
    drain_date = base_date + timedelta(hours=rng.uniform(gap * 0.5, gap))
    drain_amount = rng.uniform(50000, 300000)
    drain_source = _pick_real_nodes(g, rng, 1)
    if drain_source:
        _add_synthetic_edge(g, drain_source[0], node, drain_amount, drain_date, ring_id, "probe_and_drain")

    cashout_delay = rng.uniform(1, max(tier["timing_spread_hours"] * 0.15, 2))
    cashout_date = drain_date + timedelta(hours=cashout_delay)
    cashout_amount = drain_amount * rng.uniform(0.85, 0.97)
    cashout_target = _pick_real_nodes(g, rng, 1)
    if cashout_target:
        _add_synthetic_edge(g, node, cashout_target[0], cashout_amount, cashout_date, ring_id, "probe_and_drain")

    _add_background_noise(g, nodes, tier, base_date, ring_id, "probe_and_drain", real_amount_sample, rng)
    return nodes, [node]  # only the drop account is core; the prober alone looks innocuous


def _inject_mule_chain(g, ring_id, rng, tier, real_amount_sample):
    hops = rng.randint(*tier["size_range"])
    base_amount = rng.uniform(15000, 80000)
    base_date = _pick_base_date(g, rng)
    nodes = [_synth_node(ring_id, i) for i in range(hops)]

    real_source = _pick_real_nodes(g, rng, 1)
    real_sink = _pick_real_nodes(g, rng, 1)

    amount = base_amount
    chain_start = 0
    if real_source:
        date = _jitter_time(base_date, tier["timing_spread_hours"], rng)
        _add_synthetic_edge(g, real_source[0], nodes[0], amount, date, ring_id, "mule_chain")
        chain_start = 1

    running_date = base_date
    for i in range(chain_start, hops - 1):
        running_date = running_date + timedelta(hours=tier["timing_spread_hours"] / max(hops, 1) + rng.uniform(1, 12))
        amount = _jitter_amount(amount * rng.uniform(0.90, 0.99), tier["amount_variation"], rng)
        _add_synthetic_edge(g, nodes[i], nodes[i + 1], amount, running_date, ring_id, "mule_chain")

    if real_sink:
        running_date = running_date + timedelta(hours=rng.uniform(1, 12))
        _add_synthetic_edge(g, nodes[-1], real_sink[0], amount * rng.uniform(0.9, 0.98),
                             running_date, ring_id, "mule_chain")

    _add_background_noise(g, nodes, tier, base_date, ring_id, "mule_chain", real_amount_sample, rng)

    # only the middle 1-2 hops are core; first/last hop look like a normal pass-through
    mid = len(nodes) // 2
    core = nodes[max(0, mid - 1):mid + 1] if len(nodes) > 2 else list(nodes)
    return nodes, core


INJECTORS = {
    "cycle": _inject_cycle,
    "fan_out": _inject_fan_out,
    "fan_in": _inject_fan_in,
    "probe_and_drain": _inject_probe_and_drain,
    "mule_chain": _inject_mule_chain,
}


def _sample_real_amounts(g: nx.MultiDiGraph, rng: random.Random, n: int = 500) -> List[float]:
    amounts = [d["amount"] for _, _, d in g.edges(data=True) if not d.get("is_synthetic")]
    if not amounts:
        return []
    return rng.sample(amounts, k=min(n, len(amounts)))


def inject_synthetic_rings(
    g: nx.MultiDiGraph,
    rings_per_type_tier: int = 2,
    seed: int = 13,
    types: List[str] = None,
    tiers: List[str] = None,
) -> Tuple[nx.MultiDiGraph, List[Dict]]:
    """
    Systematic benchmark: injects `rings_per_type_tier` rings for every
    (type, tier) combination - default 5 types x 3 tiers x 2 = 30 rings.
    Mutates g in place (additive only) and returns one record per ring
    with full metadata for sliced evaluation later.
    """
    rng = random.Random(seed)
    types = types or TYPES
    tiers = tiers or list(TIERS.keys())
    real_amount_sample = _sample_real_amounts(g, rng)

    records = []
    ring_id = 0
    for ring_type in types:
        for tier_name in tiers:
            tier = TIERS[tier_name]
            for _ in range(rings_per_type_tier):
                nodes, core_nodes = INJECTORS[ring_type](g, ring_id, rng, tier, real_amount_sample)
                records.append(dict(
                    ring_id=ring_id, ring_type=ring_type, tier=tier_name,
                    nodes=nodes, core_nodes=core_nodes,
                    amount_variation=tier["amount_variation"],
                    timing_spread_hours=tier["timing_spread_hours"],
                ))
                ring_id += 1

    return g, records


if __name__ == "__main__":
    import pickle
    import sys
    sys.path.insert(0, ".")
    from graph_builder import build_multigraph

    with open("_txns_cache.pkl", "rb") as f:
        txns = pickle.load(f)

    g, report = build_multigraph(txns)
    n0, e0 = g.number_of_nodes(), g.number_of_edges()

    g, records = inject_synthetic_rings(g, rings_per_type_tier=2, seed=13)

    n1, e1 = g.number_of_nodes(), g.number_of_edges()
    print(f"Nodes: {n0} -> {n1} (+{n1 - n0})")
    print(f"Edges: {e0} -> {e1} (+{e1 - e0})")
    print(f"Rings injected: {len(records)}")

    from collections import Counter
    by_type_tier = Counter((r["ring_type"], r["tier"]) for r in records)
    for (t, tier), count in sorted(by_type_tier.items()):
        print(f"  {t:16s} {tier:7s} x{count}")

    n_core = sum(len(r["core_nodes"]) for r in records)
    n_total = sum(len(r["nodes"]) for r in records)
    print(f"\nTotal ring nodes: {n_total}, core-labeled: {n_core} ({100*n_core/n_total:.0f}%)")
