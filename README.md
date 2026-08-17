# GNN Fraud Intelligence Module

A standalone graph-neural-network layer on top of the existing bank-statement
fraud analyzer. Builds a transaction graph from your real statements, trains
a graph autoencoder on it, and scores nodes for structural anomalies —
validated against injected synthetic fraud rings since there's no ground
-truth fraud label in the dataset.

Lives entirely in this folder. Nothing in `backend/`, `frontend/`, or
`code/` (the original app, one level up) was touched.

**Read this whole file before trusting the numbers** — the two sections
"Known gaps" and "What 'fraud' means in this pipeline" matter more than the
metrics do.

## Quickstart

```bash
pip install -r requirements.txt
python run_pipeline.py --data-dir /path/to/Bank-statements-dataset
```

Takes ~1 minute on a dataset this size (84k transactions, ~4,900 graph
nodes). Writes everything to `data/processed/` — see the docstring at the
top of `run_pipeline.py` for the exact file list.

To test a single adapter directly:
```bash
python src/csv_adapter.py "some_statement.csv"
python src/excel_adapter_generic.py "some_statement.xlsx"
```

## Real results from this session

Run against the actual `Bank-statements-dataset.zip` you uploaded:

| | |
|---|---|
| Files parsed | 53 / 59 non-PDF files (11/11 CSV, 42/45 Excel) |
| Transactions ingested | 84,257 |
| Graph | 4,861 real nodes, 30,400 real edges |
| Synthetic rings injected | 12 (4× cycle, 4× fan-out, 4× fan-in), 62 nodes |
| GCN training | loss 0.71 → 0.35 over 200 epochs, ~55s, stable (not diverging) |
| Synthetic-ring recovery | **AUC 0.99**, precision@62 ≈ 65–71% (varies slightly by random seed), mean percentile rank of synthetic nodes ≈ 98.6% |

Every synthetic ring individually landed in the ~95th–100th percentile of
anomaly score. `data/processed/subgraphs/` has one rendered example per
ring type — synthetic nodes are squares, injected edges are red, node color
is anomaly score. This is the "visibly recoverable by eye" check from
Phase 3 of the plan, satisfied.

## Known gaps — and how to close them

**`excel_adapter.py` and `transaction_schema.py` were never uploaded to
this chat.** Only `counterparty_extractor.py` and `csv_adapter.py` were,
across earlier sessions. This zip therefore ships with:

- `src/transaction_schema.py` — reconstructed from the exact field names
  `csv_adapter.py` passes into `Transaction(...)`. Matches field names and
  order; any validation, defaults, or extra methods your real class has do
  not carry over.
- `src/excel_adapter_generic.py` — an independently-written Excel parser.
  It doesn't hardcode each bank's column layout; instead it scores
  candidate header rows by keyword match (date / narration / debit /
  credit / balance, plus an amount+direction-indicator fallback) and picks
  the best one per sheet. This is why it's a *generic* adapter, not *the*
  adapter — it got 42/45 real Excel files in this dataset, but it's
  necessarily less tuned than a parser written against your actual bank
  formats (which is what produced the SOA/NEFT numbers you validated
  earlier in this project).

**To close this gap:** drop your real `excel_adapter.py` and
`transaction_schema.py` into `src/`. `ingest.py` already does:
```python
try:
    from excel_adapter import read_excel_statement as read_excel_statement_impl
except ImportError:
    from excel_adapter_generic import read_excel_statement as read_excel_statement_impl
```
So your real, validated file is used automatically the moment it's present
— no other code changes needed. Re-run `run_pipeline.py` and compare
`run_summary.json`'s counts to this session's.

**Other known gaps**, none fixed here (in-scope-but-not-attempted, per
"don't touch RTGS/CASA yet" from earlier in this project):
- RTGS transactions still don't reliably produce a `counterparty_identifier`
  → they're skipped as graph edges (not fabricated), so the graph
  undercounts RTGS activity specifically.
- 3 files are genuinely legacy binary `.xls` (pre-2007 format,
  `BadZipFile` on open) — need `xlrd`, which isn't installed and there's
  no network in this sandbox to install it. Not fixable from here; should
  be trivial in an environment with internet access (`pip install xlrd`).
- 103 PDF statements aren't ingested by this pipeline at all. The
  original app already has `backend/services/pdf_reader.py` +
  `ocr_extractor_local.py` for PDF text extraction — feeding
  PDF-derived transactions into this same graph is a reasonable next
  phase, not attempted here.

## Environment constraints (why the GNN is hand-rolled numpy)

This was built and tested in a sandbox with **no network access** and
**no torch / torch_geometric / dgl installed** (checked directly, not
assumed — see `gnn_model.py` if you want to verify the reasoning).
`gnn_model.py` is a real 2-layer GCN (symmetric-normalized-adjacency
message passing, exactly per Kipf & Welling) with an inner-product
decoder, trained with hand-derived backprop + Adam — not a placeholder,
and the training curve above is the real evidence it works. If you have
torch/torch_geometric available locally, that one file's `GCNAutoencoder`
class is the only thing that would need replacing with
`torch_geometric.nn.GCNConv`; everything downstream (`evaluate.py`,
`visualize.py`) only consumes the resulting embeddings and edge scores,
so the swap is contained.

## What "fraud" means in this pipeline

There is no ground-truth fraud label anywhere in this dataset. What's
implemented and validated:

1. An unsupervised anomaly score per node (structural — how hard the GCN
   finds a node's real connections to reconstruct, combined with an
   IsolationForest outlier score over its embedding).
2. A synthetic-ring injection harness that confirms this score *can*
   separate obviously-patterned synthetic fraud (cycles, structuring,
   mule aggregation) from normal activity — AUC 0.99 on that specific
   task.

**(2) is necessary but not sufficient evidence for (1) working on real
fraud.** Synthetic rings are deliberately clean, tightly-timed, and
amount-patterned; real fraud may blend in far more with normal activity.
Treat the anomaly scores as a triage/prioritization signal for human
review, not as a fraud verdict, until validated against real labeled
cases (if/when any become available).

## PII handling

The uploaded dataset contains what look like real names, account
numbers, UPI handles, and partially-masked phone numbers/emails across
tens of thousands of transactions. The **code** here operates on that
data normally when you run it — that's the tool's job, same as the
existing app's own analytics dashboard. What's deliberately different is
what got bundled **into this zip**:

- `data/processed/node_features.csv` was **not** included as-is (real
  names/labels in the `label` column). `node_features_anonymized.csv` is
  included instead — same numeric features, `node_id` replaced with a
  short hash, `label` column dropped.
- `graph.gpickle`, `embeddings.npy`, and the full `anomaly_scores.csv`
  (whose `node_id` column carries real identifiers for non-synthetic
  nodes) are **not** included — regenerate them locally via
  `run_pipeline.py` against your own copy of the dataset; nothing about
  them requires this chat.
- The 3 subgraph PNGs use anonymized node labels (`ACC_xxxx` / `CP_xxxx`)
  — the point of the image is the structure being scored, not who's in
  it.
- `run_summary.json` and `evaluation_report.txt` are aggregate
  counts/metrics only, no per-node identifiers.
- The raw dataset itself isn't in this zip — you already have it.

## File structure

```
gnn_fraud_intelligence/
├── README.md                        this file
├── STATE.md                         phase status, in the format from your handoff kit
├── requirements.txt
├── run_pipeline.py                  end-to-end orchestrator - start here
├── src/
│   ├── transaction_schema.py        RECONSTRUCTED - see "Known gaps"
│   ├── csv_adapter.py               your real file, unchanged
│   ├── counterparty_extractor.py    your real file, unchanged (NEFT/UPI/IMPS fixes
│   │                                 from earlier sessions, incl. CBIN/SBIN/KOMA)
│   ├── excel_adapter_generic.py     RECONSTRUCTED stand-in - see "Known gaps"
│   ├── ingest.py                    batch runner, CSV+Excel, prefers a real
│   │                                 excel_adapter.py if you drop one in
│   ├── graph_builder.py             Transaction list -> networkx MultiDiGraph + adjacency
│   ├── features.py                  19 structural node features, no PII in the vectors
│   ├── labeling.py                  synthetic fraud-ring injection (cycle/fan-out/fan-in)
│   ├── gnn_model.py                 2-layer GCN graph autoencoder, hand-rolled numpy
│   ├── evaluate.py                  anomaly scoring + synthetic-ring recovery metrics
│   └── visualize.py                 anonymized subgraph rendering
└── data/
    └── processed/                   example outputs from the real run above (PII-safe
                                       subset - see "PII handling")
        ├── node_features_anonymized.csv
        ├── run_summary.json
        ├── evaluation_report.txt
        └── subgraphs/*.png
```

---

## v2 update — responding to external review

An external review of the v1 pipeline (worth reading in full if you have
it — saved in the session that produced this update) made two points
directly, and called them "the biggest improvement" and "a real
technical improvement" respectively:

1. v1's synthetic fraud rings were "deliberately clean, tightly-timed,
   and amount-patterned" — a reviewer could reasonably say "of course
   the GNN detects artificially injected, highly structured rings."
2. v1 collapsed the directed transaction graph into one binary symmetric
   adjacency matrix before the GNN ever saw it, so `A→B` and `B→A` were
   the same relationship, and "connected" was the only signal in the
   propagation matrix — money direction and amount/frequency never
   reached the message-passing step itself (only as static node features).

Both are fixed in this update — see `STATE.md` for the full technical
diff. Headline result: recovery AUC dropped from **0.994 → 0.904** going
from v1's benchmark to v2's harder one. That drop is the point: 0.90 on
noisy amounts, spread-out timing, and rings deliberately blended into
real background traffic is a far more defensible number than 0.99 on
rings designed to be easy to find.

**One honest, unresolved finding from this session**: the benchmark
table shows "hard" tier scoring as *more* detectable than "easy" tier
across all 5 fraud types — the opposite of what you'd want from a
difficulty progression. Root cause identified (ring size and
background-noise-edge count scale up together with "difficulty" in the
current tier definitions, so harder rings end up with more raw degree,
which the model picks up on independent of amount/timing subtlety) but
not yet fixed — see `STATE.md` "In progress" for the exact next step.
Flagging this here rather than only in STATE.md because it directly
affects how much to trust the difficulty ordering in
`evaluation_report.txt` until it's resolved.

### What's still open from the review
Points 2 (rule-based/IsolationForest baseline comparison — the review
calls this "probably the single most valuable experiment"), 3 (ablation
study), 6 (temporal graph analysis), 8 (held-out-pattern generalization
test), 9 (graph representation comparison), 11 (explainability), and 12
(fusion architecture) are not implemented yet. `STATE.md` has all seven
in priority order with a one-line scope note on each.

---

## v3 update — fixing the benchmark itself, not the model

A second, code-level review of v2 gave an exact 16-step technical plan
and was explicit about scope: *"your next coding task should be only
#1–5: fix the benchmark and rerun it. Don't touch the fusion, temporal
GNN, explainability, or PyTorch yet."* This update is exactly that —
`STATE.md` has the full technical diff.

**Headline result**: AUC **0.923 ± 0.013** across 5 seeds (was a single
0.904 run in v2) — properly reported with variance this time, on a
benchmark that's 5x larger (150 rings) with the tier/size confound fixed.

**A bug found in the process, not in the review**: while validating that
the rerun was trustworthy, training loss came back anomalously high
(7–8 instead of a sane ~0.7). Root cause: `features.py` log1p-transformed
the amount columns before z-scoring but not the count columns (degree,
txn_count, etc). This dataset has a real hub account with 10,875
transactions — z-scoring its raw degree next to thousands of degree-1
nodes put it 60+ standard deviations out on several feature columns,
which was destabilizing training. Fixed (all heavy-tailed columns now
log1p'd, plus a general safety clip) and verified — loss now converges
smoothly to ~0.2–0.5 instead of oscillating around 3–8. This affects the
v2 numbers reported previously: they were computed on unstable training
and should be considered superseded by this session's results, not
compared against directly as if trustworthy.

**Confirmed the tier fix worked**, not just applied it: the fraud-type ×
difficulty table now shows "hard" as the *hardest to detect* in 4 of 5
fraud types — the opposite of v2's finding (where "hard" was *always*
more detectable, in all 5 types, due to the confound). One cell
(`fan_in`/easy) still looks off, but with high variance and no
systematic pattern — read as small-sample noise, not a repeat of the
same bug. See `STATE.md` for the full honest writeup, including an
unresolved small (0.016 AUC) single-seed-vs-in-loop discrepancy that
wasn't fully root-caused.

### What's still open
Per the second review's own sequencing, the very next piece (not started
this session) is a rule-based baseline using the *existing* BankForensIQ
graph rules (circular return, burst, bidirectional flow) on the same
benchmark, plus PR-AUC and Recall@K alongside the existing AUC/Precision@K
— "the single most valuable experiment" per both reviews, deliberately
saved for the next round rather than rushed here.

---

## v4 update — the BankForensIQ rule baseline, and a real discrepancy

This round's task was explicit: inspect the actual BankForensIQ code
before writing any rule-baseline logic, and report rather than silently
reconcile any gap between what was described and what's real.

**The gap turned out to be substantial.** The described graph-
intelligence layer (Money Flow Network, Circular Return Detection,
Bidirectional Entity Flow Detection) does not exist anywhere in the
codebase — confirmed by exhaustive search, not assumed. What's real is
two single-account, time-window rules: a burst detector
(`RAPID_TRANSACTION` — >=4 debits in 10 minutes totaling >Rs 10,000) and
a smurfing-proximity signal (`is_near_threshold`, amounts in
Rs 9,000–9,999). Full writeup in `STATE.md`.

`rule_baseline.py` implements exactly those two real signals, faithfully
ported with the actual thresholds, and ships the two non-existent rules
as explicit stubs returning constant 0.0 — never invented detection
logic standing in for something that isn't there.

### Headline result

| Method | AUC | PR-AUC |
|---|---:|---:|
| Graph Rules (real BankForensIQ logic) | 0.495 ± 0.005 | 0.166 ± 0.002 |
| Isolation Forest | 0.745 ± 0.040 | 0.449 ± 0.059 |
| **GNN** | **0.867 ± 0.031** | **0.662 ± 0.057** |
| GNN + Isolation Forest | 0.928 ± 0.018 | 0.680 ± 0.045 |

The real rules score at chance (0.495 AUC) — not a small gap. Combined
with the discrepancy finding, the honest framing isn't "the GNN beats
the existing graph rules," it's **"the deployed system has no mechanism
for multi-entity structured fraud at all, and the GNN is a new
capability, not an incremental improvement on an existing one."**

**A more interesting, substantive finding** sits in the per-tier
breakdown: Isolation Forest beats the GNN by a wide margin on "easy"
difficulty (60.1% vs 12.3% PR-AUC), then the GNN pulls sharply ahead on
medium/hard (65.6% / 66.7% vs 14.8% / 7.7%). Read: obvious, tightly-
clustered synthetic rings are easy for a generic outlier detector;
once fraud blends into real background traffic, IF's simpler
embedding-distance approach degrades while the GNN's learned
reconstruction reasoning keeps working. This is a real "why it works"
answer, not just a bigger number — see `STATE.md` for the full table
and the note on where GNN+IF's naive rank-average combination visibly
underperforms GNN-alone (worth a real ablation before treating GNN+IF
as automatically the best choice).

Required disclaimer, included verbatim in `comparison_report.txt`:
*"These results evaluate recovery of synthetically injected fraud
structures. They are not measurements of confirmed real-world fraud
detection because the real dataset has no fraud ground-truth labels."*

### Tests

`tests/test_rule_baseline.py` — 7 tests, all passing: score bounds,
full node coverage, deterministic node ordering, label-membership
correctness, explicit/consistent K, and — the important ones — a
signature check proving no rule-scoring function accepts a
label/synthetic/fraud argument, and a source-level check proving
`rule_baseline.py` has no dependency on `gnn_model.py` or the GNN's
reconstruction score. The baseline is provably independent of the GNN,
not just claimed to be.

### What's still open

Ablation study, held-out-fraud-type generalization, direction/weight
ablations (does directional actually beat symmetric on a real side-by-
side, not just an architecture description), temporal analysis,
explainability, fusion (deliberately not attempted — establishing
whether the GNN adds value came first), PyTorch migration (last, as
every review has said).

---

## v5 update — corrected the v4 rule baseline

A close review of the v4 zip caught real bugs: `RAPID_BURST_THRESHOLD`
was 4 instead of the authoritative `risk_engine.py`'s 5 (that constant
was accidentally pulled from a *different* file, `unified_fraud_engine.py`
— which turns out to be a second, separately-live rule implementation
with its own disagreeing constants, not a duplicate of the same code),
the burst check incorrectly filtered to debit-only transactions, and
the "Smurfing" signal was a flat fraction-of-edges proxy standing in
for what's actually episode-level logic in the real system.

All three fixed, verified before trusting them on real data (the O(n)
burst rewrite needed for performance was checked against a brute-force
O(n²) reference across 300 random trials first), and — since the
inspection was being redone properly anyway — extended: 3 more of
`risk_engine.py`'s 7 real rules turned out to be faithfully mappable
(`HIGH_VALUE_TRANSACTION`, `SPENDING_SPIKE`, `REPEATED_TRANSACTION`) and
are now implemented too, alongside honest stubs for the 3 that are
structurally unmappable (need narration text, time-of-day, or balance
data this graph doesn't carry) and the 2 that don't exist in the
codebase at all. 12 tests now, including the exact 3 burst scenarios
requested and smurfing boundary checks.

Renamed "Graph Rules" → **"BankForensIQ Transaction Rules"** everywhere
— the inspection had already shown these aren't graph rules.

**Result under the correction, same 5-seed benchmark, nothing else
changed**: qualitatively identical to v4 — BankForensIQ's rules score
at essentially chance (AUC 0.518 ± 0.006, up from 0.495 but still
~random) even with 5 of 7 real rules faithfully reproduced instead of 2.
If anything, this makes the finding *stronger*: it's no longer possible
to attribute the weak rule score to an incomplete reproduction. The GNN
(0.862 ± 0.024) and the easy/hard crossover with Isolation Forest both
hold up unchanged. Full numbers, per-type and per-tier tables in
`STATE.md`.

**New finding this round**: this dataset has only 40 real account
nodes, and the synthetic fraud generator (untouched this session, as
required) never labels an account-type node as synthetic — every
injected ring is `node_type='counterparty'`. An account-only comparison
(the scientifically cleanest test, since the real rules were designed
for exactly that granularity) is therefore not computable under the
current benchmark — zero positive labels. Reported honestly as an open
limitation rather than worked around, since fixing it would mean
changing the fraud generator, which this round explicitly forbade.

Per this round's explicit scope: stopped here. No ablation, no
held-out-pattern test, no direction/weight ablation, no temporal, no
fusion, no PyTorch this session either.

---

## v6 update — ablation study, and a leakage finding that overturns v4/v5's own story

The mandatory inspection step (task step 1) caught something real before
any ablation code was written: the existing `isolation_scores(Z, seed)`
call inside "GNN + IF" runs Isolation Forest on **the GNN's own trained
embeddings**, not raw features. Every "Isolation Forest" number reported
in v4/v5 was secretly downstream of the GNN — not an independent
baseline. Fixed for this round's ablation only (the existing combination
is untouched, per instruction): a genuinely independent IF score on the
raw pre-GNN feature matrix, verified deterministic regardless of whether
a GNN was ever trained.

**Headline result**: GNN vs. this *true* independent IF —
ΔAUC = +0.249 ± 0.031, ΔPR-AUC = +0.444 ± 0.051, **5/5 seeds favor the
GNN on every metric**. The true IF baseline (0.615 AUC) is much weaker
than the previously-reported 0.735-0.745 — that gap was the GNN's own
embeddings doing work inside what was called "Isolation Forest." This is
the clearest evidence yet that the GNN adds real, independent detection
capability, not just an artifact of the combination.

**GNN+IF vs. GNN alone, checked properly** (not just AUC): AUC improves
reliably, but PR-AUC only wins in 2/5 seeds and Precision@K is a
coin-flip with a slightly negative mean. "GNN+IF is the best model" is
not supported once you look past AUC.

**The most interesting finding overturns v4/v5's own interpretation.**
Those versions said "IF beats GNN on easy tier, GNN wins on
medium/hard" — that pattern was an artifact of the contaminated
baseline and does not hold here. The real pattern: the GNN's own
reconstruction score is *weakest* on the most obvious ("easy") fraud
and *strongest* on medium/hard — and the existing rank-average GNN+IF
combination rescues the easy-tier weakness (39.4% vs 11.1% PR-AUC) but
*drags down* the medium/hard strength (53.1% vs 62.9%; 38.4% vs 65.2%).
The combination isn't tier-aware, and it's diluting the GNN's best
signal exactly where it doesn't need help. See
`data/processed/ablation_pr_auc_by_tier.png` for the one required
figure — it makes this pattern immediately visible.

**Not optimized** — despite this being an obvious, well-evidenced lead
for a tier-aware fusion, per the explicit instruction not to. Recorded
as the clearest concrete next step, not acted on.

6 tests, all passing, including one that actually proves independence
rather than just checking function signatures: Isolation-Forest-on-raw-
features gives a bit-identical result whether or not a GNN was ever
trained in the same process.

Per this round's stopping condition: stopped here.

---

## v7 update — held-out fraud-type generalization

The mandatory inspection caught a real contamination risk before any
code was written: the GNN's training positive-edge set includes every
edge in whatever graph it's given, real or synthetic, with no
distinction between fraud types. Naively hiding one type's labels while
leaving its structures in the training graph would not be a valid
held-out test. Checked whether the architecture even allows a proper
fix without changing it: weight matrices are shaped only by
feature/hidden/embed dimensions, independent of node count — the
architecture is inductive, so a genuine train-on-4-types /
test-on-1-type (via inference only) split is architecturally valid.
The stopping rule wasn't needed.

**Design**: separate train and test graphs sharing the same real
background but independently-injected synthetic structures (different
seed, offset +10007), with every test-graph synthetic node relabeled
`SYN:`→`SYNTEST:` to *guarantee* zero ID overlap with training — not
just make it unlikely. Verified empirically: 0 overlapping IDs across
all 25 experiments (5 fraud types × 5 seeds, 200 epochs each).

**Headline result**: the GNN beats independent Isolation Forest on
every held-out fraud type, by a wide margin every time (+9 to +51
percentage points PR-AUC) — strong evidence the GNN has learned
something more general than memorizing specific injected patterns.

**The standout finding**: comparing held-out ("unseen") performance
against Round 2's seen performance, 4 of 5 types show essentially no
degradation or even *improvement* — `fan_out` actually scores
substantially *better* held-out than seen (63.1% vs 41.4% PR-AUC).
Only `fan_in` shows a real, meaningful drop (36.2% → 25.0%) along with
by far the widest seed-to-seed spread of any type (individual-seed AUC
ranging 0.53–0.86) — a genuine, type-specific weakness, reported and
left uninvestigated per this round's explicit no-optimization rule.

Framed as **"held-out fraud-structure generalization,"** not "zero-shot
fraud detection" — per instruction, since the setup doesn't support the
stronger claim.

6 tests, all passing — including direct verification (not inference)
that train/test synthetic node and edge IDs never overlap, and that the
GNN is never trained on the test graph.

Stopped here, as instructed — no fusion, no architecture changes, no
investigation of the fan_in/fan_out anomalies.
