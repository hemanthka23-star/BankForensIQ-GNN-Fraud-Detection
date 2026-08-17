"""
A small Graph Convolutional Network trained as a Graph Autoencoder (GAE)
- Label Strategy B from STATE.md - implemented from scratch in numpy.

Why hand-rolled: torch / torch_geometric / dgl are not installed in this
sandbox and there is no network access to install them (checked - see
README "Environment constraints"). This is a real 2-layer GCN (spectral
message passing via the symmetric-normalized adjacency, exactly as in
Kipf & Welling) with an inner-product decoder, trained with manual
backprop + Adam - not a placeholder.

    P     = X @ W0                     [N, H]
    Hpre  = A_hat @ P                  [N, H]
    H1    = relu(Hpre)                 [N, H]
    M     = H1 @ W1                    [N, D]
    Z     = A_hat @ M                  [N, D]      <- node embeddings
    p_ij  = sigmoid(Z_i . Z_j)                      <- decoder, edge probability

Trained to reconstruct which node pairs are actually connected (positive
= real edges, negative = sampled non-edges). Swap-in note: if you have
torch/torch_geometric available in your own environment, the model
class below can be replaced 1:1 with a `torch_geometric.nn.GCNConv`
encoder - the rest of the pipeline (features, labeling, evaluate) is
framework-agnostic and only needs `Z` (embeddings) and `p_ij` (edge
scores) back.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


def normalize_adjacency(adj: np.ndarray) -> np.ndarray:
    """
    A_hat = D^-1/2 (A + I) D^-1/2, on a binarized version of adj (edge
    weight/count lives in node features instead - see features.py).
    """
    a = (adj > 0).astype(np.float32)
    np.fill_diagonal(a, 1.0)  # self-loops
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


class GCNAutoencoder:
    def __init__(self, n_features: int, hidden_dim: int = 64,
                 embed_dim: int = 32, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W0 = _glorot(n_features, hidden_dim, rng)
        self.W1 = _glorot(hidden_dim, embed_dim, rng)
        self.opt_W0 = AdamState.zeros_like(self.W0)
        self.opt_W1 = AdamState.zeros_like(self.W1)

    def forward(self, X: np.ndarray, A_hat: np.ndarray):
        P = X @ self.W0
        Hpre = A_hat @ P
        H1 = np.maximum(Hpre, 0.0)
        M = H1 @ self.W1
        Z = A_hat @ M
        cache = dict(X=X, A_hat=A_hat, P=P, Hpre=Hpre, H1=H1, M=M, Z=Z)
        return Z, cache

    def decode_pairs(self, Z, i_idx, j_idx):
        s = np.sum(Z[i_idx] * Z[j_idx], axis=1)
        return _sigmoid(s)

    def train_step(self, cache, i_idx, j_idx, y, lr=0.01):
        Z = cache["Z"]
        n = Z.shape[0]
        batch = len(y)

        s = np.sum(Z[i_idx] * Z[j_idx], axis=1)
        p = _sigmoid(s)
        eps = 1e-7
        loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))

        dS = (p - y) / batch  # [B]

        dZ = np.zeros_like(Z)
        np.add.at(dZ, i_idx, dS[:, None] * Z[j_idx])
        np.add.at(dZ, j_idx, dS[:, None] * Z[i_idx])

        A_hat = cache["A_hat"]
        dM = A_hat @ dZ  # A_hat symmetric, so A_hat.T == A_hat

        H1 = cache["H1"]
        dW1 = H1.T @ dM
        dH1 = dM @ self.W1.T

        dHpre = dH1 * (cache["Hpre"] > 0)
        dP = A_hat @ dHpre

        X = cache["X"]
        dW0 = X.T @ dP

        self.W0 = adam_step(self.W0, dW0, self.opt_W0, lr=lr)
        self.W1 = adam_step(self.W1, dW1, self.opt_W1, lr=lr)

        return loss

    def save(self, path):
        np.savez(path, W0=self.W0, W1=self.W1)

    @classmethod
    def load(cls, path, n_features, hidden_dim, embed_dim):
        data = np.load(path)
        model = cls(n_features, hidden_dim, embed_dim)
        model.W0, model.W1 = data["W0"], data["W1"]
        return model


def sample_negative_pairs(adj: np.ndarray, n_samples: int, rng: np.random.Generator):
    n = adj.shape[0]
    i_idx = np.empty(n_samples, dtype=np.int64)
    j_idx = np.empty(n_samples, dtype=np.int64)
    filled = 0
    while filled < n_samples:
        batch = n_samples - filled
        cand_i = rng.integers(0, n, size=batch * 2)
        cand_j = rng.integers(0, n, size=batch * 2)
        mask = (cand_i != cand_j) & (adj[cand_i, cand_j] == 0)
        take = min(mask.sum(), batch)
        idx = np.where(mask)[0][:take]
        i_idx[filled:filled + take] = cand_i[idx]
        j_idx[filled:filled + take] = cand_j[idx]
        filled += take
    return i_idx, j_idx


def train_gae(X, adj, epochs=200, lr=0.01, hidden_dim=64, embed_dim=32,
              seed=0, verbose=True):
    rng = np.random.default_rng(seed)
    A_hat = normalize_adjacency(adj)

    pos_i, pos_j = np.where(np.triu(adj, k=1) > 0)
    n_pos = len(pos_i)

    model = GCNAutoencoder(X.shape[1], hidden_dim, embed_dim, seed=seed)

    losses = []
    for epoch in range(epochs):
        neg_i, neg_j = sample_negative_pairs(adj, n_pos, rng)

        i_idx = np.concatenate([pos_i, neg_i])
        j_idx = np.concatenate([pos_j, neg_j])
        y = np.concatenate([np.ones(n_pos, dtype=np.float32),
                             np.zeros(n_pos, dtype=np.float32)])

        Z, cache = model.forward(X, A_hat)
        loss = model.train_step(cache, i_idx, j_idx, y, lr=lr)
        losses.append(loss)

        if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
            print(f"epoch {epoch:4d}  loss {loss:.4f}")

    Z_final, _ = model.forward(X, A_hat)
    return model, Z_final, A_hat, losses


if __name__ == "__main__":
    # quick smoke test on random data before touching the real graph
    rng = np.random.default_rng(0)
    n, f = 200, 10
    X = rng.normal(size=(n, f)).astype(np.float32)
    adj = (rng.random((n, n)) < 0.02).astype(np.float32)
    adj = np.triu(adj, 1)
    adj = adj + adj.T

    model, Z, A_hat, losses = train_gae(X, adj, epochs=60, verbose=True)
    print("loss[0] =", losses[0], " loss[-1] =", losses[-1])
    assert losses[-1] < losses[0], "loss did not decrease - smoke test failed"
    print("OK: loss decreased, smoke test passed")
