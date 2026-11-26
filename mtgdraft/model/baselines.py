from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def card_logit_win_rates(cards: sp.csr_matrix, won: np.ndarray, prior_games: float = 200.0) -> np.ndarray:
    """Smoothed main-deck win rate of each card, as logit relative to the overall win rate.

    This is the 17Lands "game in deck win rate" idea, shrunk toward the mean for
    rarely played cards.
    """
    in_deck = (cards > 0).astype(np.float32)
    games = np.asarray(in_deck.sum(0)).ravel()
    wins = in_deck.T @ won
    p0 = won.mean()
    wr = (wins + prior_games * p0) / (games + prior_games)
    return np.log(wr / (1 - wr)) - np.log(p0 / (1 - p0))
