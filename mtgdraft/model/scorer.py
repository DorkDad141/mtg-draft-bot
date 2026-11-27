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

    def explain(self, pack: list[str], pool: list[str], n_partners: int = 2) -> list[dict]:
        """Like rank(), but splits each score into card value and pool synergy, and names
        the pool cards that pair best and worst with it. Sorted best first."""
        pool_idx = [self._idx[n] for n in pool if n in self._idx]
        V = self.fm.V if self.fm.config.k else np.zeros((len(self.card_names), 0), np.float32)
        out = []
        for name in pack:
            i = self._idx.get(name)
            if i is None or name in self.never_pick:
                out.append({"name": name, "score": float("nan"), "value": float("nan"),
                            "synergy": float("nan"), "best": [], "worst": [],
                            "note": "basic land" if name in self.never_pick else "not in model"})
                continue
            value = float(self.fm.w_card[i])
            pair = (V[pool_idx] @ V[i]) * self.synergy_weight if pool_idx else np.zeros(0)
            syn = float(pair.sum())
            order = np.argsort(pair)
            best = [(self.card_names[pool_idx[j]], float(pair[j])) for j in order[::-1][:n_partners] if pair[j] > 0]
            worst = [(self.card_names[pool_idx[j]], float(pair[j])) for j in order[:n_partners] if pair[j] < 0]
            out.append({"name": name, "score": value + syn, "value": value, "synergy": syn,
                        "best": best, "worst": worst, "note": ""})
        return sorted(out, key=lambda r: -np.nan_to_num(r["score"], nan=-1e9))


MODELS_DIR = Path(__file__).resolve().parents[2] / "data" / "models"
SHRINK_PRIOR_DECKS = 50.0
_cache: dict[tuple, "PickScorer | None"] = {}


# The mana-aware scorer's shared settings, chosen on validation drafts of TRK, SOS and
# Powered Cube (the same values were best for all three).
MANA_TEMPERATURE = 0.025
MANA_REPLACEMENT_QUANTILE = 0.5


def load_scorer(set_code: str | None, event: str = "PremierDraft",
                models_dir: Path = MODELS_DIR, card_info: dict | None = None, kind: str = "mana"):
    """Load the trained FM for a set/event, or None if there is no model.

    With `card_info` (name -> card with colors/types/text, e.g. ArenaCardDB().by_name())
    this returns a color-aware scorer: kind "mana" (default) is the mana-aware scorer
    (pairs, triples and splashes with real mana bases, and the lane-choice model in
    lane_model.json when present); "deck" is the older two-color deck-aware scorer.
    Without card_info, the plain PickScorer.
    """
    key = (set_code, event, str(models_dir), card_info is not None, kind)
    if key not in _cache:
        path = Path(models_dir) / f"fm.{set_code}.{event}.npz"
        if set_code and path.exists():
            fm, names = FactorizationMachine.load(path)
            shrink_by_evidence(fm, names.get("meta", {}).get("card_decks"))
            if card_info is not None and kind.startswith("mana"):
                import json
                from .mana_scorer import ManaAwareScorer
                # "mana" uses lane_model.json; "mana@<file>" another lane model (for A/B tests).
                lane_path = Path(models_dir) / (kind.split("@", 1)[1] if "@" in kind else "lane_model.json")
                lm = json.loads(lane_path.read_text()) if lane_path.exists() else {}
                _cache[key] = ManaAwareScorer(fm, names["cards"], card_info, temperature=MANA_TEMPERATURE,
                                              replacement_quantile=MANA_REPLACEMENT_QUANTILE,
                                              lane_beta=lm.get("beta"), slack=lm.get("slack", 2),
                                              splash_basics=lm.get("splash_basics", 2))
            elif card_info is not None:
                from .deck_scorer import DeckAwareScorer
                _cache[key] = DeckAwareScorer(fm, names["cards"], card_info)
            else:
                _cache[key] = PickScorer(fm, names["cards"])
        elif set_code and event != "PremierDraft":
            # Quick/Traditional drafts: the decks are the same game, so the set's Premier
            # Draft model ranks them well (we only train on Premier Draft data).
            _cache[key] = load_scorer(set_code, "PremierDraft", models_dir, card_info, kind)
        else:
            _cache[key] = None
    return _cache[key]


def shrink_by_evidence(fm: FactorizationMachine, card_decks, prior_decks: float = SHRINK_PRIOR_DECKS) -> None:
    """Shrink each card's value and synergy terms toward zero by n / (n + prior), where n is
    how many training decks played it. Rare cards (a few decks) are otherwise over-trusted;
    well-observed cards are essentially unchanged."""
    if not card_decks:
        return
    n = np.asarray(card_decks, dtype=np.float64)
    f = (n / (n + prior_decks)).astype(np.float32)
    fm.w_card = fm.w_card * f
    if fm.config.k:
        fm.V = fm.V * np.sqrt(f)[:, None]


def linear_scorer(weights: np.ndarray, card_names: list[str]) -> PickScorer:
    """A context-free scorer (fixed value per card), for baselines."""
    from .fm import FMConfig
    fm = FactorizationMachine(FMConfig(k=0))
    fm.w_card = np.asarray(weights, np.float32)
    fm.V = np.zeros((len(weights), 0), np.float32)
    return PickScorer(fm, card_names, synergy_weight=0.0)
