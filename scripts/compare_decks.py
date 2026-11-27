"""How good are the bot's decks compared with human decks, including winning ones?

Everything is rated by the same independent judge as in ab_test.py (a win-rate model
trained only on held-out drafts), with each spell's uncastable share valued as a dead
card. Human drafts are taken from the training split, which the judge never saw.

Rows:
  bot (new / old scorer)       pools drafted in simulated pods, built by the deck builder
  human pools, built by us     the same deck builder applied to human final pools: the
                               fair comparison with the bots (same builder, same selection)
  human decks as played        the deck the human actually registered (17Lands data)
Human rows are also split by the drafter's match record (0-2, 3-4, 5-6, 7 wins) and
"winning record" (more match wins than losses).

Judged win rates use neutral controls (no skill or rank), so they compare decks, not
players; read them relative to each other.

    python scripts/compare_decks.py --set TRK --pods 30 --drafts 2000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import scipy.sparse as sp

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mtgdraft.carddb import ArenaCardDB  # noqa: E402
from mtgdraft.model.drafts import load_picks  # noqa: E402
from mtgdraft.model.games import load_deck_data, split_label  # noqa: E402
from mtgdraft.model.scorer import load_scorer  # noqa: E402
from scripts.ab_test import MODELS, DeckBuilder, opening_packs, run_pod, train_judge  # noqa: E402
from scripts.eval_mana import final_pools  # noqa: E402


def sig(z):
    return 1 / (1 + np.exp(-z))


def judge_actual(builder: DeckBuilder, row: sp.csr_matrix):
    """Judged logit of a real decklist, castability-adjusted. Decks rebuilt from draft data
    have no basic lands, so basics are added up to 17 lands by pip demand."""
    ms = builder.ms
    idx = np.repeat(row.indices, row.data.astype(int))
    spells = idx[ms._spell[idx]]
    lands = idx[ms._is_land[idx]]
    src = np.zeros(5)
    for i in lands:
        src += ms.mana.produces[i]
    for i in spells:
        if ms.mana.produces[i].any() and not ms._cmask[i].any():
            src += ms.mana.produces[i]
    X = row.copy().tolil()
    if len(lands) < 15:
        demand = ms.mana.single[spells].sum(0).astype(float)
        need = 17 - len(lands)
        if demand.sum():
            alloc = np.floor(need * demand / demand.sum()).astype(int)
            while alloc.sum() < need:
                alloc[int(np.argmax(np.where(demand > 0, demand / (alloc + 1), -1)))] += 1
            for x, c in enumerate("WUBRG"):
                if alloc[x] and c in builder.basic_idx:
                    X[0, builder.basic_idx[c]] += float(alloc[x])
                    src[x] += alloc[x]
    p = ms.mana.castable(spells, src[None, :], ms.slack)[0]
    penalty = float(((1 - p) * (builder.w[spells] - builder.dead)).sum())
    logit = float(builder.judge.logit(X.tocsr(), sp.csr_matrix((1, builder.n_cov), dtype=np.float32))[0])
    return logit - penalty, float(p.mean()) if len(p) else 1.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default="TRK")
    ap.add_argument("--event", default="PremierDraft")
    ap.add_argument("--pods", type=int, default=30)
    ap.add_argument("--drafts", type=int, default=2000, help="Human drafts from the training split")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    t0 = time.time()
    info = ArenaCardDB().by_name()
    A = load_scorer(args.set, args.event, card_info=info, kind="mana")
    B = load_scorer(args.set, args.event, card_info=info, kind="deck")
    names = A.card_names
    judge = train_judge(args.set, args.event, names)
    builder = DeckBuilder(judge, names, info)
    rows = {}

    # Bots
    packs = opening_packs(args.set, args.event, names)
    rng = np.random.default_rng(args.seed)
    bot = {"new": [], "old": []}
    for _ in range(args.pods):
        p24 = [packs[i] for i in rng.choice(len(packs), 24, replace=False)]
        for seating in (0, 1):
            arms = ["new" if (s + seating) % 2 == 0 else "old" for s in range(8)]
            pools = run_pod(p24, [A if a == "new" else B for a in arms])
            for arm, pool in zip(arms, pools):
                b = builder.build(pool)
                bot[arm].append((sig(b["free"]["logit"]), sig(b["human"]["logit"]), b["free"]["mean_p"]))
    print(f"{args.set}: simulated {args.pods} pods x 2 seatings  [{time.time() - t0:.0f}s]", flush=True)

    # Humans (training-split drafts: never seen by the judge)
    raw = REPO_ROOT / "data" / "raw"
    keep = lambda d: (split_label(d) == 0) & np.array([zlib.crc32(str(x).encode()) % 8 == 0 for x in d])  # noqa: E731
    picks = load_picks(raw / f"draft_data_public.{args.set}.{args.event}.csv.gz", names, keep_draft=keep,
                       cache_tag="lanefit8")
    pools = final_pools(picks, names)
    wins = picks.meta.groupby("draft_id").agg(w=("event_match_wins", "first"), l=("event_match_losses", "first"))
    ids = sorted(pools)[:args.drafts]
    data, source = load_deck_data(raw, args.set, args.event)
    first = {}
    for i, d in enumerate(data.draft_id):
        if d not in first:
            first[d] = i
    human = []
    for d in ids:
        b = builder.build(pools[d])
        actual = judge_actual(builder, data.cards[first[d]]) if d in first else (np.nan, np.nan)
        human.append((d, int(wins.loc[d, "w"]), int(wins.loc[d, "l"]), sig(b["free"]["logit"]),
                      sig(b["human"]["logit"]), b["free"]["mean_p"], sig(actual[0]), actual[1]))
    print(f"rated {len(human)} human drafts ({source} decks)  [{time.time() - t0:.0f}s]\n", flush=True)

    H = np.array([h[1:] for h in human], dtype=float)     # w, l, free, human-style, castab, actual, actual castab
    def line(label, free, hs, cast, actual=None, actual_cast=None, n=None):
        a = f"{100 * np.nanmean(actual):5.1f}%  ({np.nanmean(actual_cast):.2f})" if actual is not None else "    -          "
        print(f"  {label:<34} {100 * np.mean(free):5.1f}%   {100 * np.mean(hs):5.1f}%   ({np.mean(cast):.2f})   {a}   n={n}")
        rows[label] = {"free": float(np.mean(free)), "human_style": float(np.mean(hs)), "castability": float(np.mean(cast)),
                       "as_played": None if actual is None else float(np.nanmean(actual)), "n": n}

    print(f"  {'':<34} {'built by our builder':<29}  {'as played':<16}")
    print(f"  {'judged win rate':<34} {'free':>6}   {'human-style':>10} (castab.)  {'':>6}  (castab.)")
    for arm in ("new", "old"):
        x = np.array(bot[arm])
        line(f"bot, {arm} scorer", x[:, 0], x[:, 1], x[:, 2], n=len(x))
    groups = [("humans, all", np.ones(len(H), bool)),
              ("humans, winning record (W > L)", H[:, 0] > H[:, 1]),
              ("humans, 0-2 wins", H[:, 0] <= 2), ("humans, 3-4 wins", (H[:, 0] >= 3) & (H[:, 0] <= 4)),
              ("humans, 5-6 wins", (H[:, 0] >= 5) & (H[:, 0] <= 6)), ("humans, 7 wins (trophy)", H[:, 0] >= 7)]
    for label, m in groups:
        if m.sum():
            line(label, H[m, 2], H[m, 3], H[m, 4], H[m, 5], H[m, 6], n=int(m.sum()))
    # Per-draft human ratings, so a single deck can be placed among them (propose_deck.py).
    rows["human_decks"] = [{"wins": int(h[0]), "losses": int(h[1]), "free": float(h[2]), "human_style": float(h[3])}
                           for h in H]
    out = MODELS / f"compare_decks.{args.set}.{args.event}.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nSaved {out.relative_to(REPO_ROOT)}  [{time.time() - t0:.0f}s]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
