"""Load 17Lands draft (pick) data for offline pick evaluation.

One row per pick: the pack shown, the pool so far, the human's pick, and the
drafter's event result and skill. Card columns are aligned to a given card list
(the deck model's), so pack/pool matrices index the same cards as the model.

17Lands draft data never includes P1P1 (pick_number 0 of pack 0), so drafts have
44 picks.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .games import PROCESSED_DIR

META = ["draft_id", "pack_number", "pick_number", "pick", "pick_maindeck_rate",
        "event_match_wins", "event_match_losses", "rank",
        "user_n_games_bucket", "user_game_win_rate_bucket"]


@dataclass
class PickData:
    pack: sp.csr_matrix       # (n_picks, n_cards) copies of each card in the pack
    pool: sp.csr_matrix       # (n_picks, n_cards) copies already picked
    pick: np.ndarray          # (n_picks,) card index of the human pick (-1 if unknown)
    meta: pd.DataFrame        # per-pick metadata (META columns)

    @property
    def n(self) -> int:
        return self.pack.shape[0]


def load_picks(raw_path: str | Path, card_names: list[str], keep_draft=None,
               chunksize: int = 100_000, use_cache: bool = True, cache_tag: str = "") -> PickData:
    """Read picks, keeping only drafts where keep_draft(draft_id_array) is True."""
    raw_path = Path(raw_path)
    stat = raw_path.stat()
    key = hashlib.sha1(f"{raw_path.name}|{stat.st_size}|{stat.st_mtime_ns}|{cache_tag}|"
                       f"{'|'.join(card_names)}".encode()).hexdigest()[:10]
    cache = PROCESSED_DIR / f"{raw_path.name.removesuffix('.csv.gz')}.{cache_tag or 'all'}.{key}.npz"
    if use_cache and cache.exists():
        return _load(cache)

    idx = {n: i for i, n in enumerate(card_names)}
    header = pd.read_csv(raw_path, nrows=0).columns
    pack_cols = [c for c in header if c.startswith("pack_card_") and c[10:] in idx]
    pool_cols = [c for c in header if c.startswith("pool_") and c[5:] in idx]
    missing = {c[10:] for c in header if c.startswith("pack_card_")} - set(idx)
    if missing:
        print(f"  warning: {len(missing)} draft cards not in the deck model, ignored: {sorted(missing)[:5]}")
    pack_map = np.array([idx[c[10:]] for c in pack_cols])
    pool_map = np.array([idx[c[5:]] for c in pool_cols])
    dtypes = {c: np.int8 for c in pack_cols + pool_cols}

    packs, pools, metas = [], [], []
    n_cards = len(card_names)
    for chunk in pd.read_csv(raw_path, usecols=META + pack_cols + pool_cols, dtype=dtypes,
                             chunksize=chunksize):
        if keep_draft is not None:
            chunk = chunk[keep_draft(chunk["draft_id"].to_numpy())]
        if chunk.empty:
            continue
        for cols, cmap, out in ((pack_cols, pack_map, packs), (pool_cols, pool_map, pools)):
            m = sp.coo_matrix(chunk[cols].to_numpy(np.float32))
            out.append(sp.csr_matrix((m.data, (m.row, cmap[m.col])), shape=(len(chunk), n_cards)))
        metas.append(chunk[META].reset_index(drop=True))

    meta = pd.concat(metas, ignore_index=True)
    meta["rank"] = meta["rank"].fillna("unknown")
    data = PickData(sp.vstack(packs, format="csr"), sp.vstack(pools, format="csr"),
                    meta["pick"].map(idx).fillna(-1).astype(int).to_numpy(), meta)
    if use_cache:
        _save(cache, data)
    return data


def _save(path: Path, d: PickData) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrs = {}
    for name in ("pack", "pool"):
        m = getattr(d, name)
        arrs.update({f"{name}_data": m.data, f"{name}_indices": m.indices,
                     f"{name}_indptr": m.indptr, f"{name}_shape": np.array(m.shape)})
    meta = {c: d.meta[c].to_numpy() for c in META}
    for c in ("draft_id", "pick", "rank"):
        meta[c] = np.asarray(d.meta[c].astype(str).to_numpy(), dtype=str)
    np.savez_compressed(path, pick_idx=d.pick, **arrs, **{f"meta_{k}": v for k, v in meta.items()})


def _load(path: Path) -> PickData:
    z = np.load(path, allow_pickle=False)
    mats = {name: sp.csr_matrix((z[f"{name}_data"], z[f"{name}_indices"], z[f"{name}_indptr"]),
                                shape=tuple(z[f"{name}_shape"])) for name in ("pack", "pool")}
    meta = pd.DataFrame({c: z[f"meta_{c}"] for c in META})
    return PickData(mats["pack"], mats["pool"], z["pick_idx"], meta)
