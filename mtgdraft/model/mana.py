"""Mana model: what each card needs and makes, and how likely a deck can cast it.

Card-agnostic: everything comes from the card's mana cost, types and rules text in
Arena's card database, so it works for any set and for cube lists that change.

  pips(card)       colored mana symbols in the cost: single-color pips per color, and
                   hybrid pips ({W/U}) as color pairs. Phyrexian ({U/P}) and twobrid
                   ({2/G}) symbols can be paid without the color, so they are ignored.
  produces(card)   colors the card can make (quoted token text ignored): lands' own colors, "Add {U}" / "Add one
                   mana of any color" abilities, and fetches ("Search your library for
                   an Island or Swamp card", "a basic land card" = any color). A nonland
                   card only counts for colors it doesn't itself need (Llanowar Elves is
                   not a green source, Birds of Paradise is a source of the other four).
  castable_prob    P(a 40-card deck with S_x sources of color x has drawn enough sources
                   of every color the card needs by turn T), using the exact
                   hypergeometric distribution (the method behind Frank Karsten's land
                   counts). Colors are treated as independent.
"""

from __future__ import annotations

import re
from functools import lru_cache

import numpy as np

COLORS = "WUBRG"
CI = {c: i for i, c in enumerate(COLORS)}
BASIC_TYPES = {"Plains": "W", "Island": "U", "Swamp": "B", "Mountain": "R", "Forest": "G"}
DECK_SIZE = 40
MAX_PIPS = 4

_ADD = re.compile(r"Add ([^.]*)")
_ANY = re.compile(r"any (?:one )?colou?r|any combination of colors|of any type")
_SEARCH = re.compile(r"[Ss]earch your library for (?:a|an|up to \w+|\w+) ([^.]*?)cards?\b")


def pips(card) -> tuple[np.ndarray, dict]:
    """(single-color pips per WUBRG, {(i, j): count} hybrid pips) for a card's mana cost."""
    single = np.zeros(5, np.int64)
    hybrid: dict = {}
    if card is None or "Land" in card.types:
        return single, hybrid
    for sym in re.findall(r"\{([^}]*)\}", card.mana_cost or ""):
        parts = sym.split("/")
        if len(parts) == 1:
            if sym in CI:
                single[CI[sym]] += 1
            continue
        if any(p.isdigit() or p == "P" for p in parts):
            continue                       # twobrid or phyrexian: payable without the color
        cols = tuple(sorted(CI[p] for p in parts if p in CI))
        if len(cols) == 2:
            hybrid[cols] = hybrid.get(cols, 0) + 1
        elif len(cols) == 1:
            single[cols[0]] += 1
    return np.minimum(single, MAX_PIPS), hybrid


def produces(card) -> set[str]:
    """Colors this card can produce (see module doc)."""
    if card is None:
        return set()
    out: set[str] = set()
    land = "Land" in card.types
    # Text in quotes describes a token or a granted ability ("create a Planet token with
    # '{T}: Add one mana of any color.'"), not this card's own mana.
    text = re.sub(r'"[^"]*"', "", card.text or "")
    if land and not _ADD.search(text) and not _SEARCH.search(text):
        # No mana ability in the text (basic lands): their color identity is what they make.
        # Otherwise identity also counts colors in activation costs ({3}{B}: ...), so the
        # text decides.
        out |= set(card.color_identity)
    for clause in _ADD.findall(text):
        if "commander" in clause:
            continue                       # Arcane Signet, Command Tower: nothing in limited
        if _ANY.search(clause):
            out |= set(COLORS)
        out |= set(re.findall(r"o([WUBRG])", clause))
    for m in _SEARCH.finditer(text):
        q = m.group(1)
        if "basic land" in q:
            out |= set(COLORS)
        out |= {c for t, c in BASIC_TYPES.items() if t in q}
    if not land:
        need, hyb = pips(card)
        out -= {COLORS[i] for i in range(5) if need[i]}
    return out


@lru_cache(maxsize=1)
def hypergeom_table() -> np.ndarray:
    """T[k, S, n] = P(at least k of the S sources among n cards drawn from DECK_SIZE)."""
    from scipy.stats import hypergeom
    T = np.zeros((MAX_PIPS + 1, DECK_SIZE + 1, DECK_SIZE + 1))
    S = np.arange(DECK_SIZE + 1)[:, None]
    n = np.arange(DECK_SIZE + 1)[None, :]
    for k in range(MAX_PIPS + 1):
        T[k] = 1.0 if k == 0 else hypergeom.sf(k - 1, DECK_SIZE, S, n)
    return np.nan_to_num(T, nan=0.0)


def cards_seen(turn) -> np.ndarray:
    """Cards seen by a turn, averaging being on the play and on the draw."""
    return np.clip(np.round(6.5 + np.asarray(turn)).astype(int), 0, DECK_SIZE)


class CardMana:
    """Per-card mana arrays for a list of card names (rows align with the names)."""

    def __init__(self, names: list[str], card_info: dict):
        n = len(names)
        self.single = np.zeros((n, 5), np.int64)
        self.hyb_cols = np.full((n, 2, 2), -1, np.int64)   # up to 2 distinct hybrid pairs
        self.hyb_k = np.zeros((n, 2), np.int64)
        self.produces = np.zeros((n, 5), bool)
        self.mv = np.zeros(n, np.int64)
        self.is_land = np.zeros(n, bool)
        self.colors = [set() for _ in range(n)]           # colors needed to cast (pips and hybrids)
        for i, nm in enumerate(names):
            c = card_info.get(nm)
            if c is None:
                continue
            self.is_land[i] = "Land" in c.types
            s, h = pips(c)
            self.single[i] = s
            for j, (cols, k) in enumerate(list(h.items())[:2]):
                self.hyb_cols[i, j] = cols
                self.hyb_k[i, j] = min(k, MAX_PIPS)
            self.colors[i] = {COLORS[x] for x in range(5) if s[x]} | {COLORS[x] for cols in h for x in cols}
            for col in produces(c):
                self.produces[i, CI[col]] = True
            self.mv[i] = c.mana_value

    def castable(self, idx: np.ndarray, sources: np.ndarray, slack: float = 1.0) -> np.ndarray:
        """P(castable by turn max(mv, 1) + slack) for cards idx under each row of sources.

        sources: (L, 5) colored sources per lane. Returns (L, len(idx))."""
        T = hypergeom_table()
        L, m = sources.shape[0], len(idx)
        p = np.ones((L, m))
        if m == 0:
            return p
        S = np.clip(np.round(sources).astype(int), 0, DECK_SIZE)
        n = cards_seen(np.maximum(self.mv[idx], 1) + slack)[None, :]
        single = self.single[idx]
        for x in range(5):
            k = single[:, x]
            if k.any():
                cols = np.flatnonzero(k)
                p[:, cols] *= T[k[cols][None, :], S[:, x][:, None], n[:, cols]]
        for j in range(2):
            a, b = self.hyb_cols[idx, j, 0], self.hyb_cols[idx, j, 1]
            cols = np.flatnonzero(a >= 0)
            if len(cols):
                both = np.clip(S[:, a[cols]] + S[:, b[cols]], 0, DECK_SIZE)
                p[:, cols] *= T[self.hyb_k[idx[cols], j][None, :], both, n[:, cols]]
        p[:, self.is_land[idx]] = 1.0
        return p
