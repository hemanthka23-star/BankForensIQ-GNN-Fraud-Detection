"""
GNN v2 - direction- and edge-weight-aware Graph Autoencoder, hand-rolled
in numpy (still no torch/torch_geometric in this sandbox - see the
"environment constraints" note this project's README already has).

v1 (kept as gnn_model_v1_backup.py) collapsed the graph into ONE binary
symmetric adjacency matrix before the GNN ever saw it - so A -> B and
B -> A were the same relationship, and "connected" was the only signal
in the propagation matrix (amount/count only entered as static node
features). External review called this out directly: "money direction
matters" / "your own code explicitly does this" / "the model can learn
'A is connected to B' [instead of] 'A repeatedly sends 9500 to B via
UPI within a short period'". This version is the direct response.

Architecture
------------
Layer 1 (relational - 4 directed, weighted channels + self/residual):

    T_out_amt = A_out_amt @ X      \\
    T_out_cnt = A_out_cnt @ X       \\  A_* are row-normalized (see
    T_in_amt  = A_in_amt  @ X       /  graph_builder.build_directional_
    T_in_cnt  = A_in_cnt  @ X      /   channels) - direction- and
    T_self    = X                       weight-aware neighbor aggregation

    Mpre = T_out_amt@W1 + T_out_cnt@W2 + T_in_amt@W3 + T_in_cnt@W4 + T_self@W5
    M    = relu(Mpre)                                              [N, H]

Layer 2 (symmetric mix down to embedding space - same math as v1,
reused verbatim since direction/weight info has already been folded
into M by layer 1):

    Z = A_sym @ (M @ W6)                                           [N, D]

Decoder (asymmetric bilinear, NOT a plain dot product - so the model
can score p(i->j) != p(j->i)):

    Zq = Z @ Wq   (source/"sends to" role)
    Zk = Z @ Wk   (target/"receives from" role)
    p_ij = sigmoid(Zq_i . Zk_j)

Trained with manual backprop + Adam, same as v1 (see gnn_model_v1_
backup.py for the single-relation derivation this generalizes from -
each of the 5 layer-1 branches has an IDENTICAL gradient form to that
one, just repeated per branch and summed before the ReLU, which is
standard multi-branch backprop).
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


def normalize_symmetric(adj: np.ndarray) -> np.ndarray:
    """A_hat = D^-1/2 (A + I) D^-1/2, on a binarized adjacency - used
    only for layer 2's mixing step, same as v1."""
    a = (adj > 0).astype(np.float32)
    np.fill_diagonal(a, 1.0)
    deg = a.sum(axis=1)
    deg_inv_sqrt = np.zeros_like(deg)
    np.power(deg, -0.5, where=deg > 0, out=deg_inv_sqrt)
    d = np.diag(deg_inv_sqrt).astype(np.float32)
    return (d @ a @ d).astype(np.float32)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _glorot(fan_in, fan_out, rng):
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=(fan_in, fan_out)).astype(np.float32)


@dataclass
class AdamState:
    m: np.ndarray
    v: np.ndarray
    t: int = 0

    @classmethod
    def zeros_like(cls, arr):
        return cls(m=np.zeros_like(arr), v=np.zeros_like(arr), t=0)


def adam_step(param, grad, state: AdamState, lr=0.01, b1=0.9, b2=0.999, eps=1e-8):
    state.t += 1
    state.m = b1 * state.m + (1 - b1) * grad
    state.v = b2 * state.v + (1 - b2) * (grad ** 2)
    m_hat = state.m / (1 - b1 ** state.t)
    v_hat = state.v / (1 - b2 ** state.t)
    param -= lr * m_hat / (np.sqrt(v_hat) + eps)
    return param


CHANNELS = ("out_amt", "out_cnt", "in_amt", "in_cnt")


class DirectionalGCNAutoencoder:
    def __init__(self, n_features: int, hidden_dim: int = 64,
                 embed_dim: int = 32, seed: int = 0):
        rng = np.random.default_rng(seed)
        # layer 1: one weight matrix per channel + one self/residual
        self.W_channel = {c: _glorot(n_features, hidden_dim, rng) for c in CHANNELS}
        self.W_self = _glorot(n_features, hidden_dim, rng)
        # layer 2
        self.W2 = _glorot(hidden_dim, embed_dim, rng)
        # decoder roles
        self.Wq = _glorot(embed_dim, embed_dim, rng)
        self.Wk = _glorot(embed_dim, embed_dim, rng)

        self.opt = {
            **{c: AdamState.zeros_like(self.W_channel[c]) for c in CHANNELS},
            "self": AdamState.zeros_like(self.W_self),
            "W2": AdamState.zeros_like(self.W2),
            "Wq": AdamState.zeros_like(self.Wq),
            "Wk": AdamState.zeros_like(self.Wk),
        }

    def forward(self, X: np.ndarray, channels: Dict[str, np.ndarray], A_sym: np.ndarray):
        T = {c: channels[c] @ X for c in CHANNELS}
        T["self"] = X

        Mpre = T["self"] @ self.W_self
        for c in CHANNELS:
            Mpre = Mpre + T[c] @ self.W_channel[c]
        M = np.maximum(Mpre, 0.0)

        Z = A_sym @ (M @ self.W2)

        Zq = Z @ self.Wq
        Zk = Z @ self.Wk

        cache = dict(X=X, channels=channels, A_sym=A_sym, T=T, Mpre=Mpre, M=M, Z=Z, Zq=Zq, Zk=Zk)
        return Z, cache

    def decode_pairs(self, cache, i_idx, j_idx):
        Zq, Zk = cache["Zq"], cache["Zk"]
        s = np.sum(Zq[i_idx] * Zk[j_idx], axis=1)
        return _sigmoid(s)

    def train_step(self, cache, i_idx, j_idx, y, lr=0.01):
        Zq, Zk = cache["Zq"], cache["Zk"]
        batch = len(y)

        s = np.sum(Zq[i_idx] * Zk[j_idx], axis=1)
        p = _sigmoid(s)
        eps = 1e-7
        loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))

        dS = (p - y) / batch

        n, d = Zq.shape
        dZq = np.zeros((n, d), dtype=np.float32)
        dZk = np.zeros((n, d), dtype=np.float32)
        np.add.at(dZq, i_idx, dS[:, None] * Zk[j_idx])
        np.add.at(dZk, j_idx, dS[:, None] * Zq[i_idx])

        Z = cache["Z"]
        dWq = Z.T @ dZq
        dWk = Z.T @ dZk
        dZ = dZq @ self.Wq.T + dZk @ self.Wk.T

        A_sym = cache["A_sym"]
        M = cache["M"]
        dMW2 = A_sym @ dZ  # A_sym symmetric
        dW2 = M.T @ dMW2
        dM = dMW2 @ self.W2.T

        dMpre = dM * (cache["Mpre"] > 0)

        T = cache["T"]
        dW_self = T["self"].T @ dMpre
        dW_channel = {c: T[c].T @ dMpre for c in CHANNELS}

        self.Wq = adam_step(self.Wq, dWq, self.opt["Wq"], lr=lr)
        self.Wk = adam_step(self.Wk, dWk, self.opt["Wk"], lr=lr)
        self.W2 = adam_step(self.W2, dW2, self.opt["W2"], lr=lr)
        self.W_self = adam_step(self.W_self, dW_self, self.opt["self"], lr=lr)
        for c in CHANNELS:
            self.W_channel[c] = adam_step(self.W_channel[c], dW_channel[c], self.opt[c], lr=lr)

        return loss


def sample_negative_pairs(adj_any: np.ndarray, n_samples: int, rng: np.random.Generator):
    """adj_any: any matrix where nonzero means 'edge exists in either
    direction' - used only to avoid sampling an existing pair as a
    negative, direction of the negative itself is still random."""
    n = adj_any.shape[0]
    i_idx = np.empty(n_samples, dtype=np.int64)
    j_idx = np.empty(n_samples, dtype=np.int64)
    filled = 0
    while filled < n_samples:
        batch = n_samples - filled
        cand_i = rng.integers(0, n, size=batch * 2)
        cand_j = rng.integers(0, n, size=batch * 2)
        mask = (cand_i != cand_j) & (adj_any[cand_i, cand_j] == 0)
        take = min(mask.sum(), batch)
        idx = np.where(mask)[0][:take]
        i_idx[filled:filled + take] = cand_i[idx]
        j_idx[filled:filled + take] = cand_j[idx]
        filled += take
    return i_idx, j_idx


def train_directional_gae(X, directed_adj, symmetric_adj_any, epochs=200, lr=0.01,
                           hidden_dim=64, embed_dim=32, seed=0, verbose=True):
    """
    directed_adj      : dict from graph_builder.build_directional_channels
    symmetric_adj_any : the plain symmetric adjacency (any nonzero =
                         "connected") - used for (a) layer 2's mixing
                         matrix and (b) negative sampling and (c) as
                         the source of DIRECTED positive pairs (we
                         still need the true direction of each edge
                         for the directed decoder loss, so this fn
                         also takes the raw (i,j) directed edge list).
    """
    rng = np.random.default_rng(seed)
    A_sym = normalize_symmetric(symmetric_adj_any)

    # true directed positive pairs come from the count channel's
    # nonzero pattern *before* row-normalization is undone - simplest
    # correct source is the raw directed edge list, passed in via
    # directed_adj["_pos_i"] / ["_pos_j"] (see train.py / run_pipeline.py)
    pos_i, pos_j = directed_adj["_pos_i"], directed_adj["_pos_j"]
    n_pos = len(pos_i)

    model = DirectionalGCNAutoencoder(X.shape[1], hidden_dim, embed_dim, seed=seed)

    losses = []
    for epoch in range(epochs):
        neg_i, neg_j = sample_negative_pairs(symmetric_adj_any, n_pos, rng)

        i_idx = np.concatenate([pos_i, neg_i])
        j_idx = np.concatenate([pos_j, neg_j])
        y = np.concatenate([np.ones(n_pos, dtype=np.float32), np.zeros(n_pos, dtype=np.float32)])

        Z, cache = model.forward(X, directed_adj, A_sym)
        loss = model.train_step(cache, i_idx, j_idx, y, lr=lr)
        losses.append(loss)

        if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
            print(f"epoch {epoch:4d}  loss {loss:.4f}")

    Z_final, cache_final = model.forward(X, directed_adj, A_sym)
    return model, Z_final, cache_final, losses


if __name__ == "__main__":
    # Smoke test on random DIRECTED data - must (a) converge and
    # (b) actually be directional: p(i->j) should differ meaningfully
    # from p(j->i) on average, or the asymmetric decoder is a no-op bug.
    rng = np.random.default_rng(0)
    n, f = 200, 10
    X = rng.normal(size=(n, f)).astype(np.float32)

    directed = (rng.random((n, n)) < 0.015).astype(np.float32)
    np.fill_diagonal(directed, 0)
    pos_i, pos_j = np.where(directed > 0)

    sym_any = ((directed + directed.T) > 0).astype(np.float32)

    channels = dict(
        out_amt=directed / np.maximum(directed.sum(1, keepdims=True), 1),
        out_cnt=directed / np.maximum(directed.sum(1, keepdims=True), 1),
        in_amt=directed.T / np.maximum(directed.T.sum(1, keepdims=True), 1),
        in_cnt=directed.T / np.maximum(directed.T.sum(1, keepdims=True), 1),
        _pos_i=pos_i, _pos_j=pos_j,
    )

    model, Z, cache, losses = train_directional_gae(
        X, channels, sym_any, epochs=80, hidden_dim=32, embed_dim=16, verbose=True,
    )
    print(f"loss[0]={losses[0]:.4f}  loss[-1]={losses[-1]:.4f}")
    assert losses[-1] < losses[0], "loss did not decrease - smoke test failed"

    # directionality check: score every true edge forward vs its reverse
    p_fwd = model.decode_pairs(cache, pos_i, pos_j)
    p_rev = model.decode_pairs(cache, pos_j, pos_i)
    mean_abs_diff = np.mean(np.abs(p_fwd - p_rev))
    print(f"mean |p(i->j) - p(j->i)| over real edges = {mean_abs_diff:.4f}")
    assert mean_abs_diff > 1e-3, "decoder appears symmetric - directionality bug"
    print("OK: loss decreased AND decoder is genuinely direction-sensitive")
