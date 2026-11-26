"""Load 17Lands game data into a sparse deck matrix plus control covariates.

One row per game. Card features are main-deck copy counts (deck_<name> columns).
Covariates control for things that are not the cards: pilot skill, experience,
play/draw, mulligans and rank, so cards get credit and pilots don't.

The parsed result is cached as .npz next to the raw file's processed folder, so
the slow CSV parse happens once per download.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

META_COLS = [
    "draft_id", "draft_time", "rank", "on_play", "num_mulligans", "opp_num_mulligans",
    "won", "user_n_games_bucket", "user_game_win_rate_bucket", "main_colors",
]

# The skill bucket is the player's win rate over their 17Lands games, which likely
# includes this game. For players with few games that leaks the label, so their skill
# is treated as unknown.
SKILL_MIN_GAMES = 50


@dataclass
class GameData:
    cards: sp.csr_matrix          # (n_games, n_cards) float32 copy counts in main deck
    card_names: list[str]
    covariates: sp.csr_matrix     # (n_games, n_cov) float32 one-hot/binary controls
    covariate_names: list[str]
    won: np.ndarray               # (n_games,) float32 0/1
    draft_id: np.ndarray          # (n_games,) str
    draft_time: np.ndarray        # (n_games,) datetime64

    @property
    def n_games(self) -> int:
        return self.cards.shape[0]

    def subset(self, idx: np.ndarray) -> "GameData":
        return GameData(self.cards[idx], self.card_names, self.covariates[idx],
                        self.covariate_names, self.won[idx], self.draft_id[idx],
                        self.draft_time[idx])


def _one_hot(values: pd.Series, prefix: str) -> tuple[sp.csr_matrix, list[str]]:
    cats = pd.Categorical(values.astype(str))
    n = len(values)
    m = sp.csr_matrix((np.ones(n, np.float32), (np.arange(n), cats.codes)),
                      shape=(n, len(cats.categories)))
    return m, [f"{prefix}={c}" for c in cats.categories]


def build_covariates(meta: pd.DataFrame) -> tuple[sp.csr_matrix, list[str]]:
    n_games = meta["user_n_games_bucket"].fillna(0)
    skill = meta["user_game_win_rate_bucket"].map(lambda v: f"{v:.2f}" if pd.notna(v) else "na")
    skill = skill.where(n_games >= SKILL_MIN_GAMES, "unknown")
    parts = [
        _one_hot(skill, "skill"),
        _one_hot(n_games.astype(int), "n_games"),
        _one_hot(meta["rank"].fillna("na"), "rank"),
        _one_hot(meta["num_mulligans"].clip(upper=2), "mull"),
        _one_hot(meta["opp_num_mulligans"].clip(upper=2), "opp_mull"),
    ]
    on_play = sp.csr_matrix(meta["on_play"].astype(np.float32).to_numpy()[:, None])
    mats = [p[0] for p in parts] + [on_play]
    names = [n for p in parts for n in p[1]] + ["on_play"]
    return sp.hstack(mats, format="csr", dtype=np.float32), names


def _cache_path(raw_path: Path) -> Path:
    stat = raw_path.stat()
    key = hashlib.sha1(f"{raw_path.name}|{stat.st_size}|{stat.st_mtime_ns}|{SKILL_MIN_GAMES}".encode()).hexdigest()[:10]
    return PROCESSED_DIR / f"{raw_path.name.removesuffix('.csv.gz')}.{key}.npz"


def load_games(raw_path: str | Path, use_cache: bool = True) -> GameData:
    raw_path = Path(raw_path)
    cache = _cache_path(raw_path)
    if use_cache and cache.exists():
        return _load_cache(cache)

    header = pd.read_csv(raw_path, nrows=0).columns
    deck_cols = [c for c in header if c.startswith("deck_")]
    dtypes = {c: np.int8 for c in deck_cols}
    df = pd.read_csv(raw_path, usecols=META_COLS + deck_cols, dtype=dtypes)

    cards = sp.csr_matrix(df[deck_cols].to_numpy(np.float32))
    card_names = [c.removeprefix("deck_") for c in deck_cols]
    meta = df[META_COLS]
    cov, cov_names = build_covariates(meta)

    data = GameData(
        cards=cards, card_names=card_names, covariates=cov, covariate_names=cov_names,
        won=meta["won"].astype(np.float32).to_numpy(),
        draft_id=meta["draft_id"].astype(str).to_numpy(),
        draft_time=pd.to_datetime(meta["draft_time"]).to_numpy(),
    )
    if use_cache:
        _save_cache(cache, data)
    return data


def _save_cache(path: Path, d: GameData) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        cards_data=d.cards.data, cards_indices=d.cards.indices, cards_indptr=d.cards.indptr,
        cards_shape=np.array(d.cards.shape),
        cov_data=d.covariates.data, cov_indices=d.covariates.indices, cov_indptr=d.covariates.indptr,
        cov_shape=np.array(d.covariates.shape),
        won=d.won, draft_id=d.draft_id.astype("U32"), draft_time=d.draft_time,
        names=np.array(json.dumps({"cards": d.card_names, "cov": d.covariate_names})),
    )


def _load_cache(path: Path) -> GameData:
    z = np.load(path, allow_pickle=False)
    names = json.loads(str(z["names"]))
    return GameData(
        cards=sp.csr_matrix((z["cards_data"], z["cards_indices"], z["cards_indptr"]), shape=tuple(z["cards_shape"])),
        card_names=names["cards"],
        covariates=sp.csr_matrix((z["cov_data"], z["cov_indices"], z["cov_indptr"]), shape=tuple(z["cov_shape"])),
        covariate_names=names["cov"],
        won=z["won"], draft_id=z["draft_id"], draft_time=z["draft_time"],
    )


def split_by_draft(draft_id: np.ndarray, val_frac: float = 0.1, test_frac: float = 0.1,
                   salt: str = "mtg-draft-bot") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic train/val/test split by draft, so one deck's games stay together."""
    u = np.array([int(hashlib.md5(f"{salt}|{d}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
                  for d in draft_id])
    test = u < test_frac
    val = (u >= test_frac) & (u < test_frac + val_frac)
    train = ~(test | val)
    return np.flatnonzero(train), np.flatnonzero(val), np.flatnonzero(test)
