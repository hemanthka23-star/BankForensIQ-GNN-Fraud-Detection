# GNN Fraud Module — Project State
Last updated: 2026-08-14 (v7 session — held-out fraud-type generalization)

## What this project is
Standalone GNN-based fraud-network module, added on top of an existing
Streamlit/FastAPI bank-statement fraud analyzer
(`Automated-Banking-Fraudent-behavior--final-version-3.0v/`). Lives
entirely in `gnn_fraud_intelligence/`, never modifies the existing
`backend/`, `frontend/`, or `code/`.

## Current phase
Phase 6 of 8, v7. This session ONLY performs the held-out fraud-type
generalization study. Per explicit instruction, nothing was modified:
fraud generator, difficulty tiers, ring sizes/count, directional GNN
architecture, rule baseline, independent IF baseline, and the existing
GNN+IF combination are all byte-identical to v6. No optimization was
performed despite a clear, interesting lead (fan_in's weaker
generalization) — explicitly deferred per instruction.

## Mandatory inspection, before any code (task's own first step)
Read directly from the code, not assumed: `graph_builder.directed_edge_index()`
builds the GNN's positive training set from `set(g.edges())` with **no
distinction between real and synthetic edges, and none between fraud
types** — every edge in whatever graph is passed to
`train_directional_gae` becomes a reconstruction target. Consequence:
merely hiding one fraud type's *labels* while leaving its synthetic
structures in the training graph would NOT be a valid held-out test —
the GNN would already have been trained to reconstruct those exact
edges. This contamination is real in this codebase, not hypothetical.

However: `DirectionalGCNAutoencoder`'s weight matrices are shaped only
by feature/hidden/embed dimensions (`n_features x hidden`,
`hidden x embed`, `embed x embed`) — **none depend on node count N**.
The architecture is inductive: a model trained on one graph can be
validly applied, forward-pass only, to score a structurally different
graph. This is why the task's "critical stopping rule" (step 15) was
**not invoked** — a genuine held-out experiment is possible via a real
train/test graph split, with no architecture change.

## Exact methodology
- **TRAIN graph**: `build_multigraph(txns)` (same real background,
  always) + `inject_synthetic_rings(..., types=<4 non-held-out types>,
  seed=seed)`. GNN trained only on this graph.
- **TEST graph**: a SEPARATE `build_multigraph(txns)` call (same real
  transactions, independently rebuilt) + `inject_synthetic_rings(...,
  types=<held-out type only>, seed=seed+10007)`. Every resulting
  synthetic node is relabeled `SYN:` → `SYNTEST:` (pure string rename
  via `nx.relabel_nodes`, topology/amounts/timing untouched) —
  **guarantees** zero train/test ID overlap, not just "unlikely to
  collide". Confirmed empirically: **0 overlapping IDs across all 25
  experiments** (5 types × 5 seeds).
- Trained model scores the test graph via a single `model.forward()`
  call (inference only) — `train_directional_gae` is never called on
  the test graph (verified by source-level test).
- Feature normalization computed independently per graph (documented
  simplification — real background dominates both populations, ~4,861
  real vs ≤150 synthetic nodes either way, so stats are close in
  practice; not shared between train/test to avoid touching
  `features.py` this round).

## Tests (6, all passing)
Held-out type absent from training injection; zero train/test
synthetic node AND edge ID overlap (checked directly, not inferred);
held-out type present ONLY in test graph; seeds recorded and distinct
(train vs. train+10007); GNN trained exactly once, test graph scored
via inference only (source-level check).

## Full 25-experiment results (5 types × 5 seeds, 200 epochs each)

**GNN vs. independent Isolation Forest, held-out overall**:

| Held-out type | GNN PR-AUC | IF PR-AUC | Δ |
|---|---:|---:|---:|
| cycle | 35.1% ± 15.2% | 11.4% ± 1.0% | +23.6% |
| fan_out | 63.1% ± 23.2% | 11.8% ± 0.8% | +51.4% |
| fan_in | 25.0% ± 12.1% | 9.4% ± 0.5% | +15.6% |
| probe_and_drain | 19.1% ± 11.9% | 9.9% ± 0.6% | +9.2% |
| mule_chain | 37.6% ± 9.0% | 12.7% ± 1.4% | +24.9% |

GNN beats independent IF on every held-out type, by a wide margin every
time — the GNN generalizes to unseen fraud structures far better than a
classical baseline does even on structures it's seen.

**Per-tier** (GNN PR-AUC): full table in `data/processed/heldout_report.txt`
and the heatmap `heldout_pr_auc_heatmap.png`. Notably, for 3 of 5 types
(cycle, fan_in, mule_chain) PR-AUC *increases* from easy→hard in the
held-out setting — the opposite of the "harder = less detectable"
intuition. Plausible read, not confirmed: hard tier's extra
background-noise edges connect the fraud structure more heavily into
the *real* background the model knows well from training, which may
make the anomaly's footprint more reconstruction-salient, not less.
Flagged as an open question, not investigated further (no optimization
this round).

## Seen (Round 2) vs. unseen (held-out) — the standout finding
| Fraud type | SEEN | UNSEEN | Δ |
|---|---:|---:|---:|
| cycle | 36.4% ± 5.0% | 35.1% ± 15.2% | −1.3% |
| **fan_out** | 41.4% ± 8.9% | **63.1% ± 23.2%** | **+21.8%** |
| fan_in | 36.2% ± 6.4% | 25.0% ± 12.1% | −11.2% |
| probe_and_drain | 22.8% ± 9.8% | 19.1% ± 11.9% | −3.7% |
| mule_chain | 36.1% ± 5.8% | 37.6% ± 9.0% | +1.5% |

**Unexpected finding**: held-out performance does not collapse for
4 of 5 types — cycle and mule_chain are essentially unchanged, and
**fan_out actually scores substantially better held-out than seen**
(+21.8%, with high variance though — std 23.2%). Only **fan_in** shows
a real, meaningful drop (−11.2%) alongside the widest seed-to-seed
range of any type (individual-seed AUC from 0.53 to 0.86) — a genuine,
type-specific generalization weakness, not noise, and not investigated
further this round per the no-optimization rule.

**Not claiming "zero-shot fraud detection"** — per instruction, framed
as "held-out fraud-structure generalization": the GNN has clearly
learned something more general than memorizing 5 specific patterns
(4/5 held-out types work well, several as well as or better than
seen), but generalization is uneven across typologies, not uniform.

## Done so far (v7, this session)
- [x] Inspected training procedure before writing code (contamination
      risk identified and correctly designed around).
- [x] `src/heldout.py` — train/test graph construction, relabeling for
      guaranteed ID separation, train-once/score-via-inference split.
- [x] `run_heldout.py` — aggregation, all required tables, one heatmap
      figure, full report.
- [x] `tests/test_heldout.py` — 6 tests, all passing.
- [x] All 25 experiments run for real (5 types × 5 seeds, 200 epochs),
      checkpointed individually to disk during execution.
- [x] `data/processed/heldout_report.txt`, `heldout_results.json`,
      `heldout_pr_auc_heatmap.png` — all PII-safe (aggregate stats and
      anonymized structural node IDs only).

## In progress / next immediate step
Per this round's explicit stopping condition: **STOP here.** Next step
to be decided based on review, not assumed. Most concrete lead: why
does fan_in generalize so much worse/more variably than the other 4
types, and why does fan_out generalize *better* held-out than seen?
Neither investigated this round. Everything else queued from v6
remains untouched: tier-aware fusion, direction/weight ablation,
temporal analysis, explainability, PyTorch migration.

## Known issues / open questions
- Same as v6, unchanged (account-node gap, reconstructed
  excel_adapter.py/transaction_schema.py, RTGS, legacy `.xls`, PDFs).
- fan_in's weak/variable generalization: real, reported, unexplained.
- fan_out's better-when-held-out result: real, reported, unexplained —
  worth checking whether this replicates with more seeds/rings before
  treating it as a stable property of the pattern vs. this specific
  benchmark configuration.
- Hard-tier-more-detectable-than-easy-tier in the held-out setting for
  3/5 types: a plausible mechanism is offered above but not verified.

## File map
```
gnn_fraud_intelligence/
├── run_pipeline.py                 v3, unchanged
├── run_benchmark.py                 v3, unchanged
├── run_comparison.py                v5, unchanged
├── run_ablation.py                   v6, unchanged
├── run_heldout.py                     NEW - held-out aggregation/report
├── requirements.txt
├── src/
│   ├── transaction_schema.py        reconstructed stand-in (unchanged)
│   ├── csv_adapter.py               real, validated (unchanged)
│   ├── counterparty_extractor.py    real, validated (unchanged)
│   ├── excel_adapter_generic.py     reconstructed stand-in (unchanged)
│   ├── ingest.py                    unchanged
│   ├── graph_builder.py             unchanged
│   ├── features.py                  unchanged (NOT touched, as instructed)
│   ├── labeling.py                  unchanged (NOT touched, as instructed)
│   ├── gnn_model.py                 unchanged (NOT touched, as instructed)
│   ├── evaluate.py                  unchanged (NOT touched, as instructed)
│   ├── metrics.py                   unchanged
│   ├── rule_baseline.py             unchanged (NOT touched, as instructed)
│   ├── ablation.py                   v6, unchanged
│   ├── heldout.py                     NEW - train/test graph split + scoring
│   └── visualize.py                 unchanged
├── tests/
│   ├── test_rule_baseline.py         v5, unchanged, still passing
│   ├── test_ablation.py              v6, unchanged, still passing
│   └── test_heldout.py                NEW - 6 tests, all passing
└── data/
    └── processed/                    + NEW: heldout_report.txt, heldout_results.json,
                                        heldout_pr_auc_heatmap.png
```

## Data note
Same real `Bank-statements-dataset.zip` as v1-v6 — see README "PII
handling" for what is/isn't bundled into this zip's outputs.
