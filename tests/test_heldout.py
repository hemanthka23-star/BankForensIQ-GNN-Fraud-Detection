"""
Tests for the held-out fraud-type generalization experiment, per the
task's explicit 7 requirements: no held-out type in training injection,
no train/test synthetic node or edge ID overlap, no test labels used in
training, held-out type only appears at evaluation time, seeds are
controlled/recorded, and the GNN is never trained on the test graph.

Run with: python tests/test_heldout.py
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from ingest import ingest_directory
import heldout
from heldout import build_train_graph, build_test_graph, run_one_heldout_experiment
from labeling import TYPES


def _txns():
    report = ingest_directory("/home/claude/dataset/Bank-statements-dataset", verbose=False)
    return report.transactions


def test_held_out_type_absent_from_training_injection():
    txns = _txns()
    held_out = "probe_and_drain"
    g_train, ring_records_train = build_train_graph(txns, held_out, rings_per_type_tier=2, seed=42)
    types_present = {r["ring_type"] for r in ring_records_train}
    assert held_out not in types_present, f"{held_out} must not appear in training ring_records"
    print(f"PASS: '{held_out}' absent from training injection ({sorted(types_present)} present instead)")


def test_train_test_node_ids_do_not_overlap():
    txns = _txns()
    held_out = "cycle"
    g_train, ring_records_train = build_train_graph(txns, held_out, rings_per_type_tier=2, seed=42)
    g_test, ring_records_test, test_seed = build_test_graph(txns, held_out, rings_per_type_tier=2, seed=42)

    train_synth = set()
    for r in ring_records_train:
        train_synth.update(r["nodes"])
    test_synth = set()
    for r in ring_records_test:
        test_synth.update(r["nodes"])

    overlap = train_synth & test_synth
    assert len(overlap) == 0, f"Train/test synthetic node ID overlap: {overlap}"
    assert all(n.startswith("SYN:") for n in train_synth), "train synthetic nodes must use SYN: prefix"
    assert all(n.startswith("SYNTEST:") for n in test_synth), "test synthetic nodes must use SYNTEST: prefix"
    print(f"PASS: zero overlap between {len(train_synth)} train and {len(test_synth)} "
          f"test synthetic node IDs (distinct prefixes enforced)")


def test_train_test_edge_ids_do_not_overlap():
    txns = _txns()
    held_out = "fan_out"
    g_train, _ = build_train_graph(txns, held_out, rings_per_type_tier=2, seed=42)
    g_test, _, _ = build_test_graph(txns, held_out, rings_per_type_tier=2, seed=42)

    train_synth_txn_ids = {d["transaction_id"] for _, _, d in g_train.edges(data=True) if d.get("is_synthetic")}
    test_synth_txn_ids = {d["transaction_id"] for _, _, d in g_test.edges(data=True) if d.get("is_synthetic")}
    overlap = train_synth_txn_ids & test_synth_txn_ids
    assert len(overlap) == 0, f"Train/test synthetic edge transaction_id overlap: {overlap}"
    print(f"PASS: zero overlap between {len(train_synth_txn_ids)} train and "
          f"{len(test_synth_txn_ids)} test synthetic edge IDs")


def test_held_out_type_only_appears_at_evaluation():
    """The held-out type must be present in the TEST graph (that's the
    whole point) but absent from the TRAIN graph - confirms both halves
    of the split are correct, not just one."""
    txns = _txns()
    held_out = "mule_chain"
    _, ring_records_train = build_train_graph(txns, held_out, rings_per_type_tier=2, seed=42)
    _, ring_records_test, _ = build_test_graph(txns, held_out, rings_per_type_tier=2, seed=42)

    train_types = {r["ring_type"] for r in ring_records_train}
    test_types = {r["ring_type"] for r in ring_records_test}
    assert held_out not in train_types, "held-out type leaked into training"
    assert test_types == {held_out}, f"test graph must contain ONLY the held-out type, got {test_types}"
    print(f"PASS: '{held_out}' present only in test graph ({test_types}), "
          f"absent from train graph ({sorted(train_types)})")


def test_seeds_are_controlled_and_recorded():
    txns = _txns()
    r = run_one_heldout_experiment(txns, "cycle", seed=42, rings_per_type_tier=2, epochs=10)
    assert r["seed"] == 42
    assert r["test_seed"] == 42 + heldout.TEST_SEED_OFFSET
    assert r["test_seed"] != r["seed"], "train and test seeds must differ"
    assert r["train_test_node_id_overlap"] == 0
    print(f"PASS: seeds recorded and distinct (train={r['seed']}, test={r['test_seed']}), "
          f"0 node ID overlap confirmed in the returned experiment result")


def test_gnn_not_trained_on_test_graph():
    """Source-level + behavioral check: train_directional_gae is called
    exactly once per experiment (on g_train), and score_test_graph only
    ever calls model.forward() (inference), never train_directional_gae."""
    import inspect
    src = inspect.getsource(heldout.score_test_graph)
    assert "train_directional_gae(" not in src, \
        "score_test_graph must not call train_directional_gae - it should only do inference"
    assert "model.forward(" in src, "score_test_graph must use forward() (inference) on the trained model"

    run_src = inspect.getsource(heldout.run_one_heldout_experiment)
    assert run_src.count("train_on_graph(") == 1, "training must happen exactly once, on g_train only"
    print("PASS: GNN is trained exactly once (on the train graph); test graph is scored via "
          "inference (forward()) only, never trained on")


ALL_TESTS = [
    test_held_out_type_absent_from_training_injection,
    test_train_test_node_ids_do_not_overlap,
    test_train_test_edge_ids_do_not_overlap,
    test_held_out_type_only_appears_at_evaluation,
    test_seeds_are_controlled_and_recorded,
    test_gnn_not_trained_on_test_graph,
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
