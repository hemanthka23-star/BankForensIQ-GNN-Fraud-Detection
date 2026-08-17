"""
Shared evaluation metrics - used identically by rule_baseline.py and the
GNN evaluation path, so "Graph Rules" and "GNN" rows in the comparison
table are computed with literally the same code, not two similar-but-
different implementations.
"""

from typing import Dict, Optional

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def _precision_recall_at_k(scores: np.ndarray, labels: np.ndarray, k: int) -> Dict[str, Optional[float]]:
    if k <= 0 or k > len(scores):
        return dict(precision=None, recall=None)
    order = np.argsort(-scores)
    top_k = order[:k]
    hits = int(labels[top_k].sum())
    n_positive = int(labels.sum())
    precision = hits / k
    recall = hits / n_positive if n_positive > 0 else None
    return dict(precision=precision, recall=recall)


def compute_metrics(scores: np.ndarray, labels: np.ndarray, k: Optional[int] = None,
                     extra_ks=(10, 25, 50, 100)) -> Dict:
    """
    scores: anomaly/suspicion score per node, higher = more suspicious
    labels: boolean/0-1 array, same length, 1 = true positive (synthetic
            fraud node in this project's benchmark)
    k:      the headline K for Precision@K / Recall@K. Defaults to the
            number of positives (so Precision@K == Recall@K by
            construction) if not given - this matches how this project
            already reports it, and is printed explicitly wherever used
            so it's never ambiguous which K a number refers to.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)

    n_positive = int(labels.sum())
    n_total = len(labels)

    auc = roc_auc_score(labels, scores) if 0 < n_positive < n_total else None
    pr_auc = average_precision_score(labels, scores) if 0 < n_positive < n_total else None

    headline_k = k if k is not None else n_positive
    headline = _precision_recall_at_k(scores, labels, headline_k)

    at_k = {}
    for n in extra_ks:
        if n <= n_total:
            at_k[n] = _precision_recall_at_k(scores, labels, n)

    return dict(
        auc=auc,
        pr_auc=pr_auc,
        k=headline_k,
        precision_at_k=headline["precision"],
        recall_at_k=headline["recall"],
        precision_at_n=at_k,
        n_positive=n_positive,
        n_total=n_total,
    )


def format_metrics(name: str, m: Dict) -> str:
    lines = [f"## {name}"]
    lines.append(f"AUC          : {m['auc']:.3f}" if m["auc"] is not None else "AUC: n/a")
    lines.append(f"PR-AUC       : {m['pr_auc']:.3f}" if m["pr_auc"] is not None else "PR-AUC: n/a")
    lines.append(
        f"Precision@{m['k']:<4d}: {m['precision_at_k']:.1%}" if m["precision_at_k"] is not None else "Precision@K: n/a"
    )
    lines.append(
        f"Recall@{m['k']:<7d}: {m['recall_at_k']:.1%}" if m["recall_at_k"] is not None else "Recall@K: n/a"
    )
    for n, row in sorted(m.get("precision_at_n", {}).items()):
        p = f"{row['precision']:.1%}" if row["precision"] is not None else "n/a"
        r = f"{row['recall']:.1%}" if row["recall"] is not None else "n/a"
        lines.append(f"  Precision@{n:<4d}: {p:8s}  Recall@{n:<4d}: {r}")
    return "\n".join(lines)
