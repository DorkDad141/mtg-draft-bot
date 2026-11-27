"""Deck-structure features: the deck's curve, creature count and land count.

Card-agnostic: computed from each card's mana value and types, so the learned effect
applies to any set and to cards the model has never seen. For a deck with counts
n_k (k = spells costing <=1, 2, 3, 4, 5, 6+, creatures, lands) the features are n_k
and n_k^2 / 10. The squares let the deck model learn a best number of each (for
example "too few 2-drops" and "too many 6-drops" both cost win rate), which per-card
values cannot express. The model learns these as linear covariates, so they add 16
numbers in all, shared by every deck.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .scorer import BASIC_LANDS

GROUPS = ["mv<=1", "mv2", "mv3", "mv4", "mv5", "mv6+", "creatures", "lands"]
STRUCT_NAMES = [f"struct_{g}" for g in GROUPS] + [f"struct_{g}^2" for g in GROUPS]
SQUARE_SCALE = 10.0


def card_groups(card_names: list[str], card_info: dict) -> np.ndarray:
    """(n_cards, 8) membership of each card in the structure groups."""
    M = np.zeros((len(card_names), len(GROUPS)), np.float32)
    for i, nm in enumerate(card_names):
        c = card_info.get(nm)
        if nm in BASIC_LANDS or (c is not None and "Land" in c.types):
            M[i, 7] = 1
            continue
        if c is None:
            continue
        M[i, min(max(c.mana_value, 1), 6) - 1] = 1
        if "Creature" in c.types:
            M[i, 6] = 1
    return M


def structure_counts(cards: sp.csr_matrix, groups: np.ndarray) -> np.ndarray:
    """Deck counts per group, (n_decks, 8)."""
    return np.asarray(cards @ groups)


def structure_features(counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(counts, np.float64)
    return np.hstack([counts, counts ** 2 / SQUARE_SCALE]).astype(np.float32)


def structure_value(counts: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Logit contribution of deck structure for counts (..., 8) with learned weights w (16,)."""
    counts = np.asarray(counts, np.float64)
    return counts @ w[:8] + (counts ** 2 / SQUARE_SCALE) @ w[8:]
