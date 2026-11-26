"""Pick scorer: rank pack cards by how much they raise the deck model's win logit.

Adding card c to a deck that already holds pool x changes the factorization
machine's logit by exactly

    gain(c | x) = w_c + sum_j x_j <v_c, v_j>

i.e. the card's own value plus its synergy with every card already drafted.
`synergy_weight` scales the second term: not every pool card will make the final
deck, and a 40+ card pool is bigger than the 23 spells a deck plays.

This is the marginal-gain scorer. A later version can build the most likely final
deck from the pool and score cards against that instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from .fm import FactorizationMachine


BASIC_LANDS = frozenset({
    "Plains", "Island", "Swamp", "Mountain", "Forest", "Wastes",
    "Snow-Covered Plains", "Snow-Covered Island", "Snow-Covered Swamp",
    "Snow-Covered Mountain", "Snow-Covered Forest", "Snow-Covered Wastes",
})


@dataclass
class PickScorer:
    fm: FactorizationMachine
    card_names: list[str]
    synergy_weight: float = 1.0
    never_pick: frozenset = BASIC_LANDS   # scored NaN and ranked last (free in deckbuilding)

    def __post_init__(self):
        self._idx = {n: i for i, n in enumerate(self.card_names)}

    # ---- batch API (offline evaluation) ----------------------------------
    def score_packs(self, pool: sp.csr_matrix, pack: sp.csr_matrix) -> sp.csr_matrix:
        """Score matrix with the same sparsity as `pack`: entry (r, c) = gain of card c in pick r."""
        pack = pack.tocsr()
        rows = np.repeat(np.arange(pack.shape[0]), np.diff(pack.indptr))
        cols = pack.indices
        s = self.fm.w_card[cols].astype(np.float64)
        if self.fm.config.k and self.synergy_weight:
            P = np.asarray(pool @ self.fm.V)                 # (n_picks, k)
            s = s + self.synergy_weight * np.einsum("ij,ij->i", P[rows], self.fm.V[cols])
        return sp.csr_matrix((s, cols, pack.indptr), shape=pack.shape)

    # ---- single-pick API (live use) --------------------------------------
    def rank(self, pack: list[str], pool: list[str]) -> list[tuple[str, float]]:
        """Return pack cards sorted best first, with their win-logit gain."""
        known_pool = [self._idx[n] for n in pool if n in self._idx]
        pool_vec = np.zeros(len(self.card_names), np.float32)
        np.add.at(pool_vec, known_pool, 1.0)
        out = []
        for name in pack:
            i = self._idx.get(name)
            if i is None or name in self.never_pick:
                out.append((name, float("nan")))
                continue
            s = float(self.fm.w_card[i])
            if self.fm.config.k and self.synergy_weight:
                s += self.synergy_weight * float((pool_vec @ self.fm.V) @ self.fm.V[i])
            out.append((name, s))
        return sorted(out, key=lambda t: -np.nan_to_num(t[1], nan=-1e9))


MODELS_DIR = Path(__file__).resolve().parents[2] / "data" / "models"
_cache: dict[tuple, "PickScorer | None"] = {}


def load_scorer(set_code: str | None, event: str = "PremierDraft",
                models_dir: Path = MODELS_DIR) -> "PickScorer | None":
    """Load the trained FM for a set/event as a PickScorer, or None if there is no model."""
    key = (set_code, event, str(models_dir))
    if key not in _cache:
        path = Path(models_dir) / f"fm.{set_code}.{event}.npz"
        if set_code and path.exists():
            fm, names = FactorizationMachine.load(path)
            _cache[key] = PickScorer(fm, names["cards"])
        else:
            _cache[key] = None
    return _cache[key]


def linear_scorer(weights: np.ndarray, card_names: list[str]) -> PickScorer:
    """A context-free scorer (fixed value per card), for baselines."""
    from .fm import FMConfig
    fm = FactorizationMachine(FMConfig(k=0))
    fm.w_card = np.asarray(weights, np.float32)
    fm.V = np.zeros((len(weights), 0), np.float32)
    return PickScorer(fm, card_names, synergy_weight=0.0)
