# BankForensIQ — GNN-Based Fraud Detection

A Graph Neural Network (GNN) framework for detecting suspicious structures and potential fraud rings in banking transaction networks.

## Overview

BankForensIQ models banking activity as a transaction graph instead of treating transactions as completely independent records.

Accounts and financial entities are represented as graph nodes, while transactions create relationships between them. A direction-aware GCN autoencoder learns structural representations of the transaction network and produces anomaly scores for graph nodes.

Synthetic fraud rings are injected into the graph to evaluate how effectively the system can recover suspicious structures.

## Key Features

- Bank-statement ingestion from CSV and Excel files
- Transaction normalization
- Counterparty extraction
- Transaction graph construction
- Direction-aware graph representation
- GCN autoencoder for unsupervised anomaly detection
- Synthetic fraud-ring benchmark
- Five fraud-ring structures
- Easy, medium, and hard difficulty tiers
- Multi-seed reproducibility evaluation
- Ablation experiments
- Held-out evaluation
- Fraud-subgraph visualization
- Automated evaluation reports

## System Pipeline

```text
Bank Statements
       |
       v
Data Ingestion & Normalization
       |
       v
Transaction Extraction
       |
       v
Transaction Graph
       |
       v
Node Feature Construction
       |
       v
Direction-Aware GCN Autoencoder
       |
       v
Node Embeddings + Reconstruction Error
       |
       v
Anomaly Scores
       |
       v
Fraud-Ring Recovery & Evaluation

Graph Representation

Each account or financial entity is represented as a node in the transaction graph.

Transactions create edges between nodes, allowing the system to capture relationships and structural patterns such as:

Repeated interactions
Circular fund movement
Centralized fund collection
Centralized fund distribution
Sequential movement through intermediary accounts
Probing followed by concentrated fund movement

This graph-based representation allows suspicious network-level behavior to be detected rather than relying only on individual transaction attributes.

Fraud-Ring Benchmark

The system evaluates detection performance using synthetic fraud rings injected into the transaction graph.

The five fraud structures are:

Cycle — circular movement of funds through connected accounts
Fan-in — multiple accounts transferring funds toward a central node
Fan-out — a central node distributing funds to multiple accounts
Mule chain — sequential movement of funds through intermediary accounts
Probe-and-drain — probing activity followed by concentrated fund movement

Each fraud type is evaluated at three difficulty levels:

Easy
Medium
Hard

The benchmark uses multiple random seeds to evaluate reproducibility.

Results

The multi-seed benchmark produced:

AUC — Synthetic vs. Real: 0.923 +/- 0.013
AUC — Core Synthetic vs. Real: 0.920 +/- 0.014
Precision@k: 65.7% +/- 3.3%
Final Training Loss: 0.419 +/- 0.103

The main evaluation run produced:

AUC: 0.924
Core-only AUC: 0.917
Precision@958: 66.2%
Mean percentile — Synthetic nodes: 85.4%
Mean percentile — Real nodes: 43.0%

The multi-seed experiment used five seeds:
42, 43, 44, 45, 46

Why Use a GNN?

A transaction can appear normal when considered individually but become suspicious when its surrounding relationships are considered.

For example:
Account A
    |
    v
Account B
    |
    v
Account C
    |
    v
Account D

A graph neural network can aggregate information from neighboring nodes and learn representations that capture the surrounding transaction structure.

This makes GNNs useful for detecting structural patterns that may not be obvious from individual transaction records.

Project Structure
gnn_fraud_intelligence/
|
├── README.md
├── STATE.md
├── requirements.txt
|
├── run_pipeline.py
├── run_benchmark.py
├── run_ablation.py
├── run_comparison.py
├── run_heldout.py
|
├── src/
|   ├── ingest.py
|   ├── transaction_schema.py
|   ├── counterparty_extractor.py
|   ├── csv_adapter.py
|   ├── excel_adapter_generic.py
|   ├── graph_builder.py
|   ├── features.py
|   ├── gnn_model.py
|   ├── labeling.py
|   ├── metrics.py
|   ├── evaluate.py
|   ├── heldout.py
|   ├── ablation.py
|   ├── rule_baseline.py
|   └── visualize.py
|
└── tests/
    ├── test_ablation.py
    ├── test_heldout.py
    └── test_rule_baseline.py

Installation

Clone the repository:

git clone https://github.com/hemanthka23-star/BankForensIQ-GNN-Fraud-Detection.git
cd BankForensIQ-GNN-Fraud-Detection

Create a virtual environment:

python3 -m venv .venv
source .venv/bin/activate

Install the dependencies:

pip install -r requirements.txt

Running the Pipeline

The pipeline expects a directory containing supported bank-statement files.

python run_pipeline.py --data-dir data/dataset/Bank-statements-dataset

The pipeline performs:

Data ingestion
Transaction graph construction
Synthetic fraud-ring injection
Node feature computation
GCN autoencoder training
Anomaly scoring
Fraud-ring evaluation
Artifact generation
Experiments

Additional experiment scripts are included:
python run_benchmark.py
python run_ablation.py
python run_comparison.py
python run_heldout.py


These scripts support reproducibility benchmarking, ablation analysis, model comparison, and held-out evaluation.

Generated Artifacts

The pipeline can generate:

Anomaly scores
Node embeddings
Node features
Evaluation reports
Benchmark reports
Ablation results
Held-out evaluation results
Comparison results
Fraud-ring subgraph visualizations

Generated artifacts and raw banking data are intentionally excluded from the public repository.


Limitations

The evaluation uses synthetically injected fraud rings rather than a production-scale labeled fraud dataset.

Therefore, the reported metrics measure the model's ability to recover the injected structural patterns. They should not be interpreted as guaranteed real-world fraud-detection performance.

Real-world deployment would require:

Validated labeled fraud data
Institution-specific transaction schemas
Privacy-preserving data handling
Threshold calibration
Temporal evaluation
Distribution-shift monitoring
Human analyst investigation
Data Privacy

Raw banking statements are not included in this public repository.

The project .gitignore excludes raw financial documents and locally generated datasets from Git tracking.

Technologies : 
Python
PyTorch
NetworkX
Pandas
NumPy
Scikit-learn
Matplotlib
OpenPyXL



Authors :

Hemanth K A
Jeevan M
Tejasvi K S

PES University
B.Tech 

GitHub: https://github.com/hemanthka23-star

License

This project is intended for educational and research purposes.