"""Logistic factorization machine for deck win prediction (numpy/scipy, CPU).

logit P(win) = b + w_cov . c + w_card . x + sum_{i<j} <v_i, v_j> x_i x_j

c are the control covariates (linear only); x are main-deck card counts. Each card
has a value w_i and an embedding v_i; <v_i, v_j> is the learned synergy of the pair.
With k=0 this reduces to plain L2-regularized logistic regression.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from .metrics import log_loss


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * z))


@dataclass
class FMConfig:
    k: int = 8                 # embedding size; 0 = no pairwise terms
    l2_w: float = 1e-5         # L2 on card/covariate weights (per game, like sklearn C=1/(n*l2))
    l2_v: float = 1e-4         # L2 on embeddings
    lr: float = 0.005
    lr_decay: float = 0.8      # multiply lr by this after each epoch
    batch_size: int = 2048
    max_epochs: int = 30
    patience: int = 4          # early stopping on validation log-loss
    init_scale: float = 0.01
    seed: int = 0


@dataclass
class FactorizationMachine:
    config: FMConfig = field(default_factory=FMConfig)
    bias: float = 0.0
    w_cov: np.ndarray | None = None
    w_card: np.ndarray | None = None
    V: np.ndarray | None = None
    history: list = field(default_factory=list)

    # ---- prediction -------------------------------------------------------
    def logit(self, cards: sp.csr_matrix, cov: sp.csr_matrix) -> np.ndarray:
        z = self.bias + cov @ self.w_cov + cards @ self.w_card
        if self.config.k:
            S = cards @ self.V
            z = z + 0.5 * ((S * S).sum(1) - cards.multiply(cards) @ (self.V * self.V).sum(1))
        return np.asarray(z, dtype=np.float64).ravel()

    def predict_proba(self, cards, cov) -> np.ndarray:
        return _sigmoid(self.logit(cards, cov))

    # ---- training ---------------------------------------------------------
    def fit(self, cards: sp.csr_matrix, cov: sp.csr_matrix, y: np.ndarray,
            val: tuple | None = None, verbose: bool = True) -> "FactorizationMachine":
        c = self.config
        rng = np.random.default_rng(c.seed)
        n, n_cards = cards.shape
        n_cov = cov.shape[1]
        p = float(np.clip(y.mean(), 1e-3, 1 - 1e-3))
        self.bias = float(np.log(p / (1 - p)))
        self.w_cov = np.zeros(n_cov, np.float32)
        self.w_card = np.zeros(n_cards, np.float32)
        self.V = (rng.standard_normal((n_cards, c.k)) * c.init_scale).astype(np.float32)
        cards = cards.tocsr().astype(np.float32)
        cov = cov.tocsr().astype(np.float32)
        cards_sq = cards.multiply(cards).tocsr()

        params = ["bias", "w_cov", "w_card", "V"]
        m = {k: np.zeros_like(np.asarray(getattr(self, k), np.float32)) for k in params}
        v2 = {k: np.zeros_like(m[k]) for k in params}
        b1, b2, eps, t = 0.9, 0.999, 1e-8, 0

        best = (np.inf, None)
        bad = 0
        lr = c.lr
        for epoch in range(c.max_epochs):
            order = rng.permutation(n)
            for start in range(0, n, c.batch_size):
                idx = order[start:start + c.batch_size]
                X, Xc, X2 = cards[idx], cov[idx], cards_sq[idx]
                z = self.bias + Xc @ self.w_cov + X @ self.w_card
                if c.k:
                    S = X @ self.V
                    z = z + 0.5 * ((S * S).sum(1) - X2 @ (self.V * self.V).sum(1))
                e = (_sigmoid(z) - y[idx]) / len(idx)
                grads = {
                    "bias": np.float32(e.sum()),
                    "w_cov": Xc.T @ e + c.l2_w * self.w_cov,
                    "w_card": X.T @ e + c.l2_w * self.w_card,
                }
                if c.k:
                    grads["V"] = X.T @ (e[:, None] * S) - self.V * (X2.T @ e)[:, None] + c.l2_v * self.V
                else:
                    grads["V"] = np.zeros_like(self.V)
                t += 1
                for k in params:
                    g = np.asarray(grads[k], np.float32)
                    m[k] = b1 * m[k] + (1 - b1) * g
                    v2[k] = b2 * v2[k] + (1 - b2) * g * g
                    step = lr * (m[k] / (1 - b1 ** t)) / (np.sqrt(v2[k] / (1 - b2 ** t)) + eps)
                    setattr(self, k, getattr(self, k) - step)
            self.bias = float(self.bias)
            lr *= c.lr_decay

            train_ll = log_loss(y, self.predict_proba(cards, cov))
            val_ll = log_loss(val[2], self.predict_proba(val[0], val[1])) if val else np.nan
            self.history.append((epoch, train_ll, val_ll))
            if verbose:
                print(f"    epoch {epoch:2d}  train {train_ll:.5f}  val {val_ll:.5f}", flush=True)
            if val is None:
                continue
            if val_ll < best[0] - 1e-5:
                best, bad = (val_ll, self._state()), 0
            else:
                bad += 1
                if bad >= c.patience:
                    break
        if best[1] is not None:
            self._set_state(best[1])
        return self

    def _state(self):
        return (self.bias, self.w_cov.copy(), self.w_card.copy(), self.V.copy())

    def _set_state(self, s):
        self.bias, self.w_cov, self.w_card, self.V = s

    # ---- persistence ------------------------------------------------------
    def save(self, path: str | Path, card_names: list[str], covariate_names: list[str], **meta) -> None:
        import json
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, bias=self.bias, w_cov=self.w_cov, w_card=self.w_card, V=self.V,
                            names=np.array(json.dumps({"cards": card_names, "cov": covariate_names,
                                                       "config": self.config.__dict__, "meta": meta})))

    @classmethod
    def load(cls, path: str | Path) -> tuple["FactorizationMachine", dict]:
        import json
        z = np.load(path, allow_pickle=False)
        names = json.loads(str(z["names"]))
        fm = cls(FMConfig(**names["config"]))
        fm.bias, fm.w_cov, fm.w_card, fm.V = float(z["bias"]), z["w_cov"], z["w_card"], z["V"]
        return fm, names
