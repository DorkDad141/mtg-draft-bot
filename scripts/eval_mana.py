"""Calibrate the mana model on what humans actually built.

17Lands game data records each deck's main colors and splash colors. For held-out
drafts, take the drafter's final pool, ask the mana-aware scorer which lane (pair,
triple, or pair + splash) it would build, and compare with the deck they played:

  main     the scorer's main colors equal the human's main colors
  shape    same shape: 2 colors / 2 + splash / 3 colors (/ other)
  splash   the scorer splashes exactly when the human did

This checks only the mana model's knobs (slack, splash basics, dead and replacement
levels); it never looks at individual picks. Sets need game data (Cube, SOS; not TRK).

    python scripts/eval_mana.py --set Cube_-_Powered --drafts 1500
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mtgdraft.carddb import ArenaCardDB  # noqa: E402
from mtgdraft.model.drafts import load_picks  # noqa: E402
from mtgdraft.model.fm import FactorizationMachine  # noqa: E402
from mtgdraft.model.games import split_label  # noqa: E402
from mtgdraft.model.mana_scorer import ManaAwareScorer  # noqa: E402
from mtgdraft.model.scorer import shrink_by_evidence  # noqa: E402

ORDER = "WUBRG"


def norm(s) -> str:
    s = "" if not isinstance(s, str) else s
    return "".join(c for c in ORDER if c in s)


def shape(main: str, splash: str) -> str:
    if len(main) == 2:
        return "2+splash" if splash else "2"
    if len(main) == 3:
        return "3+splash" if splash else "3"
    return f"{len(main)}" + ("+splash" if splash else "")


def human_decks(raw: Path, set_code: str, event: str, drafts: set) -> pd.DataFrame:
    path = raw / f"game_data_public.{set_code}.{event}.csv.gz"
    out = []
    for ch in pd.read_csv(path, usecols=["draft_id", "build_index", "main_colors", "splash_colors"], chunksize=200000):
        out.append(ch[ch["draft_id"].isin(drafts)])
    g = pd.concat(out)
    g = g[g["build_index"] == g.groupby("draft_id")["build_index"].transform("min")]   # first build
    g = g.groupby("draft_id").agg(main=("main_colors", "first"), splash=("splash_colors", "first"))
    g["main"] = g["main"].map(norm)
    g["splash"] = g["splash"].map(norm)
    return g


def final_pools(picks, card_names) -> dict:
    meta = picks.meta
    last = meta.groupby("draft_id").apply(lambda g: g.index[np.argmax(g["pack_number"] * 100 + g["pick_number"])],
                                          include_groups=False)
    pools = {}
    for d, r in last.items():
        row = picks.pool[r]
        pool = [card_names[i] for i, c in zip(row.indices, row.data) for _ in range(int(c))]
        if picks.pick[r] >= 0:
            pool.append(card_names[picks.pick[r]])
        pools[d] = pool
    return pools


def score(scorer, pools: dict, humans: pd.DataFrame):
    main_ok = shape_ok = splash_ok = 0
    bot_shapes, human_shapes = Counter(), Counter()
    n = 0
    for d, pool in pools.items():
        if d not in humans.index:
            continue
        hm, hs = humans.loc[d, "main"], humans.loc[d, "splash"]
        bm, bs = scorer.best_lane(pool)
        main_ok += bm == hm
        shape_ok += shape(bm, bs) == shape(hm, hs)
        splash_ok += bool(bs) == bool(hs)
        bot_shapes[shape(bm, bs)] += 1
        human_shapes[shape(hm, hs)] += 1
        n += 1
    return n, main_ok / n, shape_ok / n, splash_ok / n, bot_shapes, human_shapes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default="Cube_-_Powered")
    ap.add_argument("--event", default="PremierDraft")
    ap.add_argument("--drafts", type=int, default=1500, help="Held-out drafts per split")
    ap.add_argument("--slack", type=float, nargs="+", default=[0, 1, 2])
    ap.add_argument("--splash-basics", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--dead", type=float, nargs="+", default=[0.05])
    ap.add_argument("--quantile", type=float, nargs="+", default=[0.7])
    ap.add_argument("--strength", default="deck", choices=["deck", "above"])
    args = ap.parse_args()

    t0 = time.time()
    raw, models = REPO_ROOT / "data" / "raw", REPO_ROOT / "data" / "models"
    fm, names = FactorizationMachine.load(models / f"fm.{args.set}.{args.event}.npz")
    shrink_by_evidence(fm, names.get("meta", {}).get("card_decks"))
    card_names = names["cards"]
    info = ArenaCardDB().by_name()
    picks = load_picks(raw / f"draft_data_public.{args.set}.{args.event}.csv.gz", card_names,
                       keep_draft=lambda d: split_label(d) > 0, cache_tag="valtest")
    lab = split_label(picks.meta["draft_id"].to_numpy())
    results = {}
    best = None
    for split_name, sid in (("val", 1), ("test", 2)):
        ids = sorted(set(picks.meta["draft_id"][lab == sid]))[:args.drafts]
        keep = np.flatnonzero(picks.meta["draft_id"].isin(set(ids)).to_numpy())
        from mtgdraft.model.drafts import PickData
        sub = PickData(picks.pack[keep], picks.pool[keep], picks.pick[keep], picks.meta.iloc[keep].reset_index(drop=True))
        pools = final_pools(sub, card_names)
        humans = human_decks(raw, args.set, args.event, set(ids))
        print(f"\n{split_name}: {len(pools)} drafts, {len(humans)} with recorded decks  [{time.time() - t0:.0f}s]")
        grid = (list(itertools.product(args.slack, args.splash_basics, args.dead, args.quantile))
                if split_name == "val" else [best])
        best_key = -1
        for sl, sb, dq, q in grid:
            sc = ManaAwareScorer(fm, card_names, info, slack=sl, splash_basics=sb, dead_quantile=dq,
                                 replacement_quantile=q, strength_mode=args.strength)
            n, m_ok, s_ok, sp_ok, bot, hum = score(sc, pools, humans)
            tot = sum(hum.values())
            print(f"  slack {sl:g} splash_basics {sb} dead {dq:g} q {q:g}: main colors {m_ok:.1%}  "
                  f"shape {s_ok:.1%}  splash yes/no {sp_ok:.1%} | bot shapes "
                  + " ".join(f"{k} {v / n:.0%}" for k, v in sorted(bot.items())), flush=True)
            if split_name == "val" and m_ok + s_ok > best_key:
                best_key, best = m_ok + s_ok, (sl, sb, dq, q)
        print("  human shapes " + " ".join(f"{k} {v / tot:.0%}" for k, v in sorted(hum.items())))
        results[split_name] = best
    print(f"\nChosen on val: slack {best[0]:g}, splash_basics {best[1]}, dead {best[2]:g}, q {best[3]:g}  "
          f"[{time.time() - t0:.0f}s]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
