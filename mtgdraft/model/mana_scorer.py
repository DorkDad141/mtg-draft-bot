"""Mana-aware pick scorer: the deck-aware scorer, with real mana bases.

Lanes are the decks the pool might become: the 10 two-color pairs, the 10 three-color
triples, and each pair with a splash of one of the other three colors (50 lanes).

For each lane, a 17-land mana base is built from the pool:
  * a nonbasic land is played if it makes at least two of the lane's colors (a dual,
    tri-land or fetch for this deck), or if its own value in the lane beats a basic land's
    (utility lands; only when the model learned from decklists that include basics, i.e.
    17Lands game data, since draft-derived decks have no basics to compare with); a land
    making just one lane color is only a worse basic, so it is left out. Nonland mana sources (Signets, Moxen, dorks, ramp) whose cost fits the lane
    also count. Each counts as a source of every lane color it makes;
  * the remaining basics go first to the splash color (up to `splash_basics`, minus
    the fixing the pool already has for it), then to the main colors in proportion
    to how many pips of each color the pool's lane cards need.

Each card's value in a lane is then
    eff = p * val + (1 - p) * dead
where p is the chance the mana base can cast it by turn mana value + `slack`
(exact hypergeometric, see mana.py), val is its learned deck value in the lane (as in
the deck-aware scorer: card value plus its synergy with the lane's basic lands), and
dead is the value of a near-useless card (a low quantile of the set's values). So
off-color cards are worth about nothing, a single-pip bomb in a splash color keeps
much of its value, a double-pip card in a splash keeps little, and adding a splash
weakens the main colors' castability: the real cost of splashing.

Lane strength, weights, thresholds and the candidate's gain work as in the deck-aware
scorer, using eff instead of val. In addition, a candidate that makes colored mana is
also scored by how much its fixing raises each lane's strength (the lane rebuilt with
it in the mana base), so duals, fetches, Signets and Moxen are worth a lot to a pool
that needs them and little to one that doesn't.

Nothing here is per-card hand tuning: castability is exact probability, and the few
knobs (slack, splash_basics, quantiles, temperature) are shared by every card and
tuned on held-out drafts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import scipy.sparse as sp

from .fm import FactorizationMachine
from .mana import COLORS, CardMana
from .scorer import BASIC_LANDS

BASIC_OF = {"W": "Plains", "U": "Island", "B": "Swamp", "R": "Mountain", "G": "Forest"}


def make_lanes(triples: bool = True, splashes: bool = True) -> list[tuple[str, str]]:
    """[(main colors, splash color or "")]."""
    lanes = [("".join(p), "") for p in combinations(COLORS, 2)]
    if triples:
        lanes += [("".join(t), "") for t in combinations(COLORS, 3)]
    if splashes:
        lanes += [("".join(p), s) for p in combinations(COLORS, 2) for s in COLORS if s not in p]
    if triples and splashes:
        lanes += [("".join(t), s) for t in combinations(COLORS, 3) for s in COLORS if s not in t]
    return lanes


# Lane-choice model features (card-agnostic; see lane_features).
LANE_FEATURES = ["value", "depth", "minor_depth", "triple", "splash", "splash_cards"]


def lane_name(main: str, splash: str) -> str:
    return main + (f"+{splash.lower()}" if splash else "")


@dataclass
class ManaAwareScorer:
    fm: FactorizationMachine
    card_names: list[str]
    card_info: dict
    synergy_weight: float = 1.0
    temperature: float = 0.05
    replacement_quantile: float = 0.7
    dead_quantile: float = 0.05
    slack: float = 1.0                 # turns of grace for casting (splash cards are rarely needed on curve)
    splash_basics: int = 3             # most basics given to a splash color
    lands: int = 17
    deck_slots: int = 23
    softness: float = 0.01
    tiebreak: float = 0.005
    triples: bool = True
    lane_beta: object = None           # lane-choice model weights (LANE_FEATURES); None: softmax of strength
    land_utility: bool = True          # play lands for their own value (only meaningful when the model
                                       # learned from decklists that include basic lands: game data)
    strength_mode: str = "deck"        # "deck": value of the best 23 over a shared baseline; "above": over each lane's bar
    splashes: bool = True
    never_pick: frozenset = BASIC_LANDS
    _pre: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        names = self.card_names
        n = len(names)
        self._idx = {nm: i for i, nm in enumerate(names)}
        k = self.fm.config.k
        self._V = (self.fm.V if k else np.zeros((n, 0), np.float32)).astype(np.float64)
        self._w = self.fm.w_card.astype(np.float64)
        self.mana = m = CardMana(names, self.card_info)
        self.missing_info = [nm for nm in names if nm not in self.card_info]
        self._is_basic = np.array([nm in BASIC_LANDS for nm in names])
        self._is_land = m.is_land | self._is_basic
        self._spell = ~self._is_land
        self.lanes_def = make_lanes(self.triples, self.splashes)
        L = len(self.lanes_def)
        self._lane_mask = np.zeros((L, 5), bool)     # main + splash colors
        self._main_mask = np.zeros((L, 5), bool)
        self._splash = np.full(L, -1)
        for li, (main, spl) in enumerate(self.lanes_def):
            for c in main:
                self._main_mask[li, COLORS.index(c)] = True
            if spl:
                self._splash[li] = COLORS.index(spl)
        self._lane_mask = self._main_mask.copy()
        has = self._splash >= 0
        self._lane_mask[np.flatnonzero(has), self._splash[has]] = True

        # Card colors as a mask, for "fits the lane" checks.
        self._cmask = np.zeros((n, 5), bool)
        for i, cs in enumerate(m.colors):
            for c in cs:
                self._cmask[i, COLORS.index(c)] = True
        # Lane values with synergy to the lane's (default) basics, replacement and dead levels.
        val = np.zeros((L, n))
        r = np.zeros(L)
        dead = np.zeros(L)
        land_thr = np.zeros(L)
        for li in range(L):
            bvec = np.zeros(n)
            mains = np.flatnonzero(self._main_mask[li])
            for x in mains:
                j = self._idx.get(BASIC_OF[COLORS[x]])
                if j is not None:
                    bvec[j] = (self.lands - (self.splash_basics if self._splash[li] >= 0 else 0)) / len(mains)
            if self._splash[li] >= 0:
                j = self._idx.get(BASIC_OF[COLORS[self._splash[li]]])
                if j is not None:
                    bvec[j] = self.splash_basics
            val[li] = self._w + self.synergy_weight * (self._V @ (bvec @ self._V))
            fits = self._spell & ~(self._cmask & ~self._main_mask[li]).any(1)
            vals = val[li][fits] if fits.any() else val[li][self._spell]
            r[li] = np.quantile(vals, self.replacement_quantile)
            dead[li] = np.quantile(vals, self.dead_quantile)
            basics_here = [self._idx[BASIC_OF[COLORS[x]]] for x in mains if BASIC_OF[COLORS[x]] in self._idx]
            land_thr[li] = max(val[li][basics_here].mean() if basics_here else 0.0, r[li])
        self._val, self._r, self._dead, self._land_thr = val, r, dead, land_thr
        # One shared baseline for comparing lanes: a near-useless card's value, set-wide.
        self._base = float(np.quantile(self._w[self._spell], self.dead_quantile)) if self._spell.any() else 0.0
        self._sd = float(np.std(self._w[self._spell])) if self._spell.any() else 1.0
        if self.strength_mode == "deck":
            self._dead = np.minimum(self._dead, self._base)   # an uncastable card adds nothing to a deck

    # ---- mana base ---------------------------------------------------------------
    def _sources(self, pool_idx: np.ndarray, extra: int | None = None) -> np.ndarray:
        """(L, 5) colored sources of each lane's mana base built from the pool (+ an extra card)."""
        idx = pool_idx if extra is None else np.append(pool_idx, extra)
        L = len(self.lanes_def)
        use_land, use_rock, lane_prod = self._mana_cards(idx)
        src = (lane_prod & (use_land | use_rock)[:, :, None]).sum(1).astype(float)   # (L, 5)
        n_lands = np.minimum(use_land.sum(1), self.lands - 2)
        basics = (self.lands - n_lands).astype(float)
        has = self._splash >= 0
        sp_col = np.where(has, self._splash, 0)
        sp_have = src[np.arange(L), sp_col]
        sp_b = np.where(has, np.clip(self.splash_basics - sp_have, 0, None), 0.0)
        sp_b = np.minimum(sp_b, basics)
        src[np.arange(L), sp_col] += sp_b
        rest = basics - sp_b
        spells = self._spell[idx]
        demand = (self.mana.single[idx][spells].sum(0)[None, :] + 1.0) * self._main_mask       # (L, 5)
        src += rest[:, None] * demand / demand.sum(1, keepdims=True)
        return src

    def _mana_cards(self, idx: np.ndarray):
        """Which pool cards each lane's deck plays for mana: (use_land, use_rock, lane_prod).

        A nonbasic land is played if it makes 2+ of the lane's colors, or if its own value in
        the lane beats a basic (utility lands). This is the one land rule, used both when
        ranking picks and when building the deck."""
        prod = self.mana.produces[idx]                                       # (m, 5)
        land = self._is_land[idx] & ~self._is_basic[idx]
        lane_prod = prod[None, :, :] & self._lane_mask[:, None, :]          # (L, m, 5)
        n_cols = lane_prod.sum(2)                                            # (L, m)
        fits = ~(self._cmask[idx][None, :, :] & ~self._lane_mask[:, None, :]).any(2)
        utility = (self._val[:, idx] > self._land_thr[:, None]) if self.land_utility else np.zeros((len(self.lanes_def), len(idx)), bool)
        use_land = land[None, :] & ((n_cols >= 2) | utility)
        use_rock = (n_cols >= 1) & ~land[None, :] & fits & self._spell[idx][None, :]
        return use_land, use_rock, lane_prod

    def build_deck(self, pool: list[str], lane: tuple[str, str] | None = None) -> dict:
        """The deck this scorer projects for the pool: the lane it rates most likely (or the
        given one), the best 23 spells by castability-weighted value, the nonbasic lands its
        land rule plays, and basics by the same mana base (splash first, then pip demand)."""
        pool_idx = np.array([self._idx[n] for n in pool if n in self._idx], dtype=int)
        src = self._sources(pool_idx)
        deck = self._deck(pool_idx, src)
        strength, _, _, spells, chosen, p = deck
        if lane is not None:
            li = self.lanes_def.index(lane)
        else:
            feats = self.lane_features(pool_idx, src, deck) if self.lane_beta is not None else None
            li = int(np.argmax(self._weights(strength, feats)))
        picked = spells[chosen[li]] if len(spells) else np.zeros(0, int)
        p_of = dict(zip(spells.tolist(), p[li].tolist())) if len(spells) else {}
        use_land, _, _ = self._mana_cards(pool_idx)
        lands = list(pool_idx[use_land[li]])[: self.lands - 2]
        n_basic = self.lands - len(lands)
        # Basics: the same split as the mana base (splash color first, then pip demand).
        have = self.mana.produces[lands].astype(float).sum(0) if lands else np.zeros(5)
        alloc = np.zeros(5)
        sp = self._splash[li]
        if sp >= 0:
            alloc[sp] = min(max(self.splash_basics - have[sp], 0), n_basic)
        demand = (self.mana.single[picked].sum(0) + 1.0) * self._main_mask[li] if len(picked) else \
            self._main_mask[li].astype(float)
        rest = n_basic - alloc.sum()
        share = rest * demand / demand.sum()
        alloc += np.floor(share)
        for x in np.argsort(-(share - np.floor(share)))[: int(round(n_basic - alloc.sum()))]:
            alloc[x] += 1
        basics = {BASIC_OF[COLORS[x]]: int(alloc[x]) for x in range(5) if alloc[x] > 0}
        return {"lane": self.lanes_def[li], "lane_name": lane_name(*self.lanes_def[li]),
                "spells": [self.card_names[i] for i in picked],
                "castable": [float(p_of.get(int(i), 1.0)) for i in picked],
                "lands": [self.card_names[i] for i in lands], "basics": basics}

    def _deck(self, pool_idx: np.ndarray, src: np.ndarray):
        """Per lane: strength, threshold, synergy sum of the projected deck, and its cards."""
        spells = pool_idx[self._spell[pool_idx]]
        L = len(self.lanes_def)
        thr = self._r.copy()
        strength = np.zeros(L)
        vsum = np.zeros((L, self._V.shape[1]))
        lands = pool_idx[self._is_land[pool_idx] & ~self._is_basic[pool_idx]]
        land_v = self._V[lands].sum(0) if len(lands) else 0.0
        chosen_mask = np.zeros((L, len(spells)), bool)
        p = self.mana.castable(spells, src, self.slack)
        if len(spells):
            eff = p * self._val[:, spells] + (1 - p) * self._dead[:, None]
            order = np.argsort(-eff, axis=1)[:, :self.deck_slots]
            rows = np.arange(L)[:, None]
            chosen_mask[rows, order] = True
            top = eff[rows, order]
            if self.strength_mode == "deck":
                strength = np.clip(top - self._base, 0, None).sum(1)
            else:
                strength = np.clip(top - self._r[:, None], 0, None).sum(1)
            if len(spells) >= self.deck_slots:
                thr = np.maximum(self._r, top[:, -1])
            vsum = chosen_mask.astype(float) @ self._V[spells]
        vsum = vsum + land_v
        return strength, thr, vsum, spells, chosen_mask, p

    def _weights(self, strength, feats=None):
        if self.lane_beta is not None and feats is not None:
            z = feats @ np.asarray(self.lane_beta) / max(self.temperature, 1e-9) * 0.05
        else:
            z = strength / self.temperature
        w = np.exp(z - z.max())
        return w / w.sum()

    def lane_features(self, pool_idx: np.ndarray, src=None, deck=None) -> np.ndarray:
        """(L, F) features of each lane for the lane-choice model, in set-independent units.

        value         castability-weighted value of the best 23 spells above the shared
                      baseline, in standard deviations of the set's card values
        depth         expected castable playables (value above baseline) in the pool / 23,
                      not capped at 23, so deeper lanes keep counting
        minor_depth   the same for the lane's thinnest main color (a "pair" with almost
                      no cards of one color is really mono-color)
        triple        three main colors
        splash        has a splash color
        splash_cards  expected castable playables needing the splash color / 5
        """
        if src is None:
            src = self._sources(pool_idx)
        if deck is None:
            deck = self._deck(pool_idx, src)
        strength, _, _, spells, chosen, p = deck
        L = len(self.lanes_def)
        F = np.zeros((L, len(LANE_FEATURES)))
        F[:, 0] = strength / self._sd
        F[:, 3] = self._main_mask.sum(1) == 3
        F[:, 4] = self._splash >= 0
        if len(spells):
            good = (self._val[:, spells] > self._base)                 # (L, m) playable-quality cards
            pg = p * good
            F[:, 1] = pg.sum(1) / self.deck_slots
            need = self._cmask[spells]                                 # (m, 5)
            per_color = pg @ need                                      # (L, 5): playables needing each color
            main_depth = np.where(self._main_mask, per_color, np.inf)
            F[:, 2] = main_depth.min(1) / self.deck_slots
            has = self._splash >= 0
            F[has, 5] = per_color[np.flatnonzero(has), self._splash[has]] / 5.0
        return F

    # ---- scoring -------------------------------------------------------------------
    def _scores(self, cand: np.ndarray, pool_idx: np.ndarray):
        src = self._sources(pool_idx)
        deck = self._deck(pool_idx, src)
        strength, thr, vsum, _, _, _ = deck
        wts = self._weights(strength, self.lane_features(pool_idx, src, deck) if self.lane_beta is not None else None)
        p = self.mana.castable(cand, src, self.slack)                       # (L, c)
        syn = self.synergy_weight * (vsum @ self._V[cand].T)                # (L, c)
        base = p * (self._val[:, cand] + syn) + (1 - p) * self._dead[:, None]
        cut = np.where(self._is_land[cand][None, :], self._land_thr[:, None], thr[:, None])
        x = (base - cut) / self.softness
        gain = self.softness * np.logaddexp(0.0, x)
        # Lands outside the lane's colors (and not colorless utility) do nothing there.
        land_c = self._is_land[cand]
        if land_c.any():
            prod = self.mana.produces[cand]
            useless = land_c[None, :] & (prod.any(1)[None, :] & ~(prod[None, :, :] & self._lane_mask[:, None, :]).any(2))
            gain = np.where(useless, 0.0, gain)
        # Fixing: how much the card raises each lane's strength as a mana source.
        fix = np.zeros_like(gain)
        makers = np.flatnonzero(self.mana.produces[cand].any(1) & ~self._is_basic[cand])
        for j in makers:
            s2 = self._sources(pool_idx, extra=cand[j])
            if np.allclose(s2, src):
                continue
            st2 = self._deck(pool_idx, s2)[0]
            fix[:, j] = np.clip(st2 - strength, 0, None)
        score = wts @ (gain + fix)
        pool_syn = self.synergy_weight * (self._V[pool_idx].sum(0) @ self._V[cand].T) if len(pool_idx) \
            else np.zeros(len(cand))
        raw = self._w[cand] + pool_syn
        fit = wts @ p
        return score + self.tiebreak * raw, raw, pool_syn, fit, wts, gain + fix

    # ---- public API (same shape as the other scorers) -------------------------------
    def lanes(self, pool: list[str], top: int = 3) -> list[tuple[str, float, int]]:
        """Most likely decks for the pool: (lane, weight, likely-castable cards in its projected deck)."""
        pool_idx = np.array([self._idx[n] for n in pool if n in self._idx], dtype=int)
        src = self._sources(pool_idx)
        deck = self._deck(pool_idx, src)
        strength, _, _, spells, chosen, p = deck
        wts = self._weights(strength, self.lane_features(pool_idx, src, deck) if self.lane_beta is not None else None)
        order = np.argsort(-wts)[:top]
        return [(lane_name(*self.lanes_def[li]), float(wts[li]),
                 int(((p[li] >= 0.5) & chosen[li]).sum()) if len(spells) else 0) for li in order]

    def rank(self, pack: list[str], pool: list[str]) -> list[tuple[str, float]]:
        return [(r["name"], r["score"]) for r in self.explain(pack, pool)]

    def explain(self, pack: list[str], pool: list[str], n_partners: int = 2) -> list[dict]:
        pool_idx = np.array([self._idx[n] for n in pool if n in self._idx], dtype=int)
        known = [(nm, self._idx[nm]) for nm in pack if nm in self._idx and nm not in self.never_pick]
        out = []
        if known:
            cand = np.array([i for _, i in known], dtype=int)
            score, raw, pool_syn, fit, wts, lane_gain = self._scores(cand, pool_idx)
            for j, (nm, i) in enumerate(known):
                pair = (self._V[pool_idx] @ self._V[i]) * self.synergy_weight if len(pool_idx) else np.zeros(0)
                order = np.argsort(pair)
                best = [(self.card_names[pool_idx[t]], float(pair[t])) for t in order[::-1][:n_partners] if pair[t] > 0]
                worst = [(self.card_names[pool_idx[t]], float(pair[t])) for t in order[:n_partners] if pair[t] < 0]
                contrib = wts * lane_gain[:, j]
                li = int(np.argmax(contrib))
                out.append({"name": nm, "score": float(score[j]), "value": float(self._w[i]),
                            "synergy": float(pool_syn[j]), "fit": float(fit[j]),
                            "best": best, "worst": worst, "note": "",
                            "lane": lane_name(*self.lanes_def[li]) if contrib[li] > 0 else ""})
        for nm in pack:
            if nm not in self._idx or nm in self.never_pick:
                out.append({"name": nm, "score": float("nan"), "value": float("nan"), "synergy": float("nan"),
                            "fit": float("nan"), "best": [], "worst": [], "lane": "",
                            "note": "basic land" if nm in self.never_pick else "not in model"})
        return sorted(out, key=lambda r: -np.nan_to_num(r["score"], nan=-1e9))

    def best_lane(self, pool: list[str]) -> tuple[str, str]:
        """The single most likely (main colors, splash) for a finished pool."""
        pool_idx = np.array([self._idx[n] for n in pool if n in self._idx], dtype=int)
        src = self._sources(pool_idx)
        deck = self._deck(pool_idx, src)
        if self.lane_beta is not None:
            return self.lanes_def[int(np.argmax(self.lane_features(pool_idx, src, deck) @ np.asarray(self.lane_beta)))]
        return self.lanes_def[int(np.argmax(deck[0]))]

    def castable_share(self, pool: list[str], lane: tuple[str, str]) -> float:
        """Share of the pool's nonland cards castable (p >= 0.5) in the given lane's deck."""
        pool_idx = np.array([self._idx[n] for n in pool if n in self._idx], dtype=int)
        spells = pool_idx[self._spell[pool_idx]]
        if not len(spells):
            return float("nan")
        li = self.lanes_def.index(lane)
        p = self.mana.castable(spells, self._sources(pool_idx)[li:li + 1], self.slack)[0]
        return float((p >= 0.5).mean())

    def score_packs(self, pool: sp.csr_matrix, pack: sp.csr_matrix) -> sp.csr_matrix:
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
