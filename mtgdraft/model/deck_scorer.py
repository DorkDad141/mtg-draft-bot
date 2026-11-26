"""Deck-aware pick scorer: values cards by how much they improve the deck the pool
is likely to become, instead of treating every pool card as part of the deck.

For each two-color pair P:
  * castable(P): nonland cards whose colors are within P (colorless always), and
    lands that make at least one of P's colors (or colorless utility lands).
  * val_P(c) = w_c + alpha * <v_c, sum of P's basic lands>: the card's value in a
    deck of those colors. Basic lands are model features, so their learned pairings
    give the model's own sense of color fit.
  * The projected deck D_P is the best 23 castable nonland pool cards by val_P.
    Slots the pool can't fill yet are filled by "replacement-level" cards whose value
    r_P is a quantile of val_P over all castable cards in the set.
  * threshold_P is the value of the weakest card the deck would cut to make room:
    r_P while the deck has open slots, else the weakest card in D_P.
  * strength_P = sum over D_P of max(0, val_P - r_P); pair weights are
    softmax(strength / tau). Early in the draft all pairs are about equal; as the
    pool commits, one or two pairs dominate.

A candidate c then scores
    sum_P weight_P * softplus(val_P(c) + alpha * <v_c, sum of D_P> - threshold_P)
for castable pairs (0 for the others). Softplus is ~max(0, x) but keeps a little
value for on-color cards just below the cut, so they still beat off-color cards.
Off-color cards score about zero, and earlier picks that don't fit the direction
stop counting. Mono-color cards fit 4 of the 10 pairs and gold cards only 1, so early
picks naturally favour flexible cards. A small fraction (`tiebreak`) of the plain
marginal score orders cards that would all score zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import scipy.sparse as sp

from .fm import FactorizationMachine
from .scorer import BASIC_LANDS

COLORS = "WUBRG"
BASIC_OF = {"W": "Plains", "U": "Island", "B": "Swamp", "R": "Mountain", "G": "Forest"}
PAIRS = ["".join(p) for p in combinations(COLORS, 2)]


@dataclass
class DeckAwareScorer:
    fm: FactorizationMachine
    card_names: list[str]
    card_info: dict                    # name -> object with .colors, .color_identity, .types
    synergy_weight: float = 1.0
    temperature: float = 0.05          # softmax temperature over pair strengths (logit units, tuned)
    replacement_quantile: float = 0.7  # replacement level = this quantile of castable card values (tuned)
    basics_per_color: int = 8
    deck_slots: int = 23
    softness: float = 0.01             # softplus width (logit): below-cutoff on-color cards keep a little value
    tiebreak: float = 0.005
    never_pick: frozenset = BASIC_LANDS
    _pre: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        names = self.card_names
        n = len(names)
        self._idx = {nm: i for i, nm in enumerate(names)}
        k = self.fm.config.k
        V = self.fm.V if k else np.zeros((n, 0), np.float32)
        self._V = V.astype(np.float64)
        w = self.fm.w_card.astype(np.float64)
        self._w = w

        colors = []
        is_land = np.zeros(n, bool)
        is_basic = np.array([nm in BASIC_LANDS for nm in names])
        missing = []
        for i, nm in enumerate(names):
            c = self.card_info.get(nm)
            if c is None:
                missing.append(nm)
                colors.append(set())
                continue
            land = "Land" in c.types
            is_land[i] = land
            colors.append(set(c.color_identity if land else c.colors))
        self.missing_info = missing
        self._is_land, self._is_basic = is_land, is_basic

        castable = np.zeros((len(PAIRS), n), bool)
        val = np.zeros((len(PAIRS), n))
        r = np.zeros(len(PAIRS))
        land_thr = np.zeros(len(PAIRS))
        basic_sum = np.zeros((len(PAIRS), V.shape[1]))
        for p, pair in enumerate(PAIRS):
            ps = set(pair)
            for i in range(n):
                cs = colors[i]
                if is_basic[i]:
                    castable[p, i] = False
                elif is_land[i]:
                    castable[p, i] = (not cs) or bool(cs & ps)
                else:
                    castable[p, i] = cs <= ps
            bvec = np.zeros(n)
            for col in pair:
                j = self._idx.get(BASIC_OF[col])
                if j is not None:
                    bvec[j] = self.basics_per_color
            basic_sum[p] = bvec @ self._V
            val[p] = w + self.synergy_weight * (self._V @ basic_sum[p])
            spells = castable[p] & ~is_land
            r[p] = np.quantile(val[p][spells], self.replacement_quantile) if spells.any() else 0.0
            basics_here = [self._idx[BASIC_OF[c]] for c in pair if BASIC_OF[c] in self._idx]
            # A nonbasic land replaces a basic, but it is also a pick that could have been a
            # playable spell, so it must beat replacement level too.
            land_thr[p] = max(val[p][basics_here].mean() if basics_here else 0.0, r[p])
        self._castable, self._val, self._r, self._land_thr = castable, val, r, land_thr

    # ---- core ------------------------------------------------------------------
    def _lanes(self, pool_idx: np.ndarray):
        """Per pair: weight, threshold, embedding sum of the projected deck, deck card ids."""
        P = len(PAIRS)
        strength = np.zeros(P)
        thr = self._r.copy()
        deck_vsum = np.zeros((P, self._V.shape[1]))
        decks = []
        for p in range(P):
            ok = self._castable[p, pool_idx] if len(pool_idx) else np.zeros(0, bool)
            cand = pool_idx[ok]
            spells = cand[~self._is_land[cand]]
            lands = cand[self._is_land[cand]]
            vals = self._val[p, spells]
            order = np.argsort(-vals)
            chosen = spells[order[:self.deck_slots]]
            if len(chosen) >= self.deck_slots:
                thr[p] = max(self._r[p], self._val[p, chosen].min())
            strength[p] = np.clip(self._val[p, chosen] - self._r[p], 0, None).sum()
            deck_cards = np.concatenate([chosen, lands])
            deck_vsum[p] = self._V[deck_cards].sum(0) if len(deck_cards) else 0.0
            decks.append(chosen)
        z = strength / self.temperature
        wts = np.exp(z - z.max())
        wts /= wts.sum()
        return wts, thr, deck_vsum, decks, strength

    def _scores(self, cand: np.ndarray, pool_idx: np.ndarray):
        wts, thr, deck_vsum, decks, _ = self._lanes(pool_idx)
        V = self._V[cand]                                        # (c, k)
        syn = self.synergy_weight * (deck_vsum @ V.T)            # (P, c)
        base = self._val[:, cand] + syn
        cut = np.where(self._is_land[cand][None, :], self._land_thr[:, None], thr[:, None])
        x = (base - cut) / self.softness
        soft = self.softness * np.logaddexp(0.0, x)              # softplus: ~max(0, base - cut), but smooth
        gain = np.where(self._castable[:, cand], soft, 0.0)
        score = wts @ gain
        pool_syn = self.synergy_weight * (self._V[pool_idx].sum(0) @ V.T) if len(pool_idx) else np.zeros(len(cand))
        raw = self._w[cand] + pool_syn
        fit = wts @ self._castable[:, cand]
        return score + self.tiebreak * raw, raw, pool_syn, fit, wts

    # ---- public API (same shape as PickScorer) ------------------------------------
    def lanes(self, pool: list[str], top: int = 3) -> list[tuple[str, float, int]]:
        """Most likely color pairs for the pool: (pair, weight, playables in the projected deck)."""
        pool_idx = np.array([self._idx[n] for n in pool if n in self._idx], dtype=int)
        wts, _, _, decks, _ = self._lanes(pool_idx)
        order = np.argsort(-wts)[:top]
        return [(PAIRS[p], float(wts[p]), int(len(decks[p]))) for p in order]

    def rank(self, pack: list[str], pool: list[str]) -> list[tuple[str, float]]:
        return [(r["name"], r["score"]) for r in self.explain(pack, pool)]

    def explain(self, pack: list[str], pool: list[str], n_partners: int = 2) -> list[dict]:
        pool_idx = np.array([self._idx[n] for n in pool if n in self._idx], dtype=int)
        known = [(nm, self._idx[nm]) for nm in pack if nm in self._idx and nm not in self.never_pick]
        out = []
        if known:
            cand = np.array([i for _, i in known], dtype=int)
            score, raw, pool_syn, fit, wts = self._scores(cand, pool_idx)
            for j, (nm, i) in enumerate(known):
                pair = (self._V[pool_idx] @ self._V[i]) * self.synergy_weight if len(pool_idx) else np.zeros(0)
                order = np.argsort(pair)
                best = [(self.card_names[pool_idx[t]], float(pair[t])) for t in order[::-1][:n_partners] if pair[t] > 0]
                worst = [(self.card_names[pool_idx[t]], float(pair[t])) for t in order[:n_partners] if pair[t] < 0]
                lane_gain = self._castable[:, i] * wts
                out.append({"name": nm, "score": float(score[j]), "value": float(self._w[i]),
                            "synergy": float(pool_syn[j]), "fit": float(fit[j]),
                            "best": best, "worst": worst, "note": "",
                            "lane": PAIRS[int(np.argmax(lane_gain))] if lane_gain.any() else ""})
        for nm in pack:
            if nm not in self._idx or nm in self.never_pick:
                out.append({"name": nm, "score": float("nan"), "value": float("nan"), "synergy": float("nan"),
                            "fit": float("nan"), "best": [], "worst": [], "lane": "",
                            "note": "basic land" if nm in self.never_pick else "not in model"})
        return sorted(out, key=lambda r: -np.nan_to_num(r["score"], nan=-1e9))

    def score_packs(self, pool: sp.csr_matrix, pack: sp.csr_matrix) -> sp.csr_matrix:
        """Batch scoring for offline evaluation; same output shape as PickScorer.score_packs."""
        pack = pack.tocsr()
        pool = pool.tocsr()
        data = np.empty(pack.nnz)
        never = np.array([nm in self.never_pick for nm in self.card_names])
        for r in range(pack.shape[0]):
            a, b = pack.indptr[r], pack.indptr[r + 1]
            cand = pack.indices[a:b]
            pa, pb = pool.indptr[r], pool.indptr[r + 1]
            pool_idx = np.repeat(pool.indices[pa:pb], pool.data[pa:pb].astype(int))
            s = self._scores(cand, pool_idx)[0]
            s[never[cand]] = -1e9
            data[a:b] = s
        return sp.csr_matrix((data, pack.indices, pack.indptr), shape=pack.shape)
