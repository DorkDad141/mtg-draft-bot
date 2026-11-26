"""Offline pick evaluation on held-out 17Lands drafts.

Replays every pick of the validation and test drafts (the same draft split the
deck model was trained with, so no held-out draft was seen in training) and asks
each scorer to rank the pack the human saw, given the human's pool so far.

Two kinds of evidence:
  agreement   How often the scorer's top card is the human's pick, and the human
              pick's percentile in the scorer's ranking. Reported for all drafters
              and for strong drafters (17Lands win rate >= 0.60 over 50+ games).
  outcome     Per draft, the share of picks where the human took the scorer's top
              card, regressed on the draft's match wins with skill, experience
              and rank controls. A positive coefficient means drafts that
              followed the scorer more won more, beyond what the drafter's skill
              explains. Observational, so read it as supporting evidence.

Example:
    python scripts/eval_picks.py --set Cube_-_Powered --event PremierDraft
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mtgdraft.model.baselines import card_logit_win_rates  # noqa: E402
from mtgdraft.model.drafts import load_picks  # noqa: E402
from mtgdraft.model.fm import FactorizationMachine  # noqa: E402
from mtgdraft.model.games import load_deck_data, split_label  # noqa: E402
from mtgdraft.model.deck_scorer import PAIRS, DeckAwareScorer  # noqa: E402
from mtgdraft.model.scorer import PickScorer, linear_scorer  # noqa: E402

STRONG_WR, STRONG_GAMES = 0.60, 50


def pick_stats(scores: sp.csr_matrix, pick: np.ndarray):
    """Per pick: is the human pick the top score, and its percentile among the other cards."""
    n = scores.shape[0]
    rows = np.repeat(np.arange(n), np.diff(scores.indptr))
    pick_score = np.asarray(scores[np.arange(n), pick]).ravel()
    s = scores.data
    above = np.bincount(rows, weights=(s > pick_score[rows]), minlength=n)
    tied = np.bincount(rows, weights=(s == pick_score[rows]), minlength=n) - 1
    others = np.diff(scores.indptr) - 1
    top1 = (above == 0) & (tied == 0)
    pct = 1 - (above + 0.5 * tied) / np.maximum(others, 1)
    return top1, pct


def outcome_regression(meta: pd.DataFrame, agree: np.ndarray):
    """OLS of draft match wins on agreement share + controls. Returns (coef, se) for agreement."""
    df = meta[["draft_id", "event_match_wins", "user_game_win_rate_bucket",
               "user_n_games_bucket", "rank"]].copy()
    df["agree"] = agree
    d = df.groupby("draft_id").agg(wins=("event_match_wins", "first"), agree=("agree", "mean"),
                                   skill=("user_game_win_rate_bucket", "first"),
                                   ngames=("user_n_games_bucket", "first"), rank=("rank", "first"))
    skill = np.where(d["ngames"] >= STRONG_GAMES, d["skill"].astype(str), "unknown")
    X = pd.concat([pd.Series(1.0, index=d.index, name="const"), d["agree"],
                   pd.get_dummies(skill, prefix="s", drop_first=True).set_index(d.index),
                   pd.get_dummies(d["ngames"].astype(str), prefix="n", drop_first=True),
                   pd.get_dummies(d["rank"].astype(str), prefix="r", drop_first=True)], axis=1).astype(float)
    y = d["wins"].astype(float).to_numpy()
    Xv = X.to_numpy()
    beta, *_ = np.linalg.lstsq(Xv, y, rcond=None)
    resid = y - Xv @ beta
    XtX_inv = np.linalg.pinv(Xv.T @ Xv)
    meat = (Xv * resid[:, None]).T @ (Xv * resid[:, None])     # HC0 robust SE
    cov = XtX_inv @ meat @ XtX_inv
    j = list(X.columns).index("agree")
    return float(beta[j]), float(np.sqrt(cov[j, j])), len(d)


def evaluate(name, scorer, picks, strong, stage, results, verbose=True):
    scores = scorer.score_packs(picks.pool, picks.pack)
    top1, pct = pick_stats(scores, picks.pick)
    coef, se, n_drafts = outcome_regression(picks.meta, top1)
    r = {"scorer": name, "top1": top1.mean(), "pct": pct.mean(),
         "top1_strong": top1[strong].mean(), "pct_strong": pct[strong].mean(),
         "wins_per_agree": coef, "wins_per_agree_se": se, "n_drafts": n_drafts}
    for st in ("P1 early", "P1 late", "P2-P3"):
        r[f"top1_{st}"] = top1[stage == st].mean()
    results.append(r)
    if verbose:
        print(f"  {name:<24} top1 {r['top1']:.3f}  pct {r['pct']:.3f} | strong top1 {r['top1_strong']:.3f} "
              f"pct {r['pct_strong']:.3f} | by stage {r['top1_P1 early']:.3f}/{r['top1_P1 late']:.3f}/"
              f"{r['top1_P2-P3']:.3f} | wins/agree {coef:+.2f} +/- {se:.2f}", flush=True)
    return r


def pool_fit(names: list[str], info: dict) -> float:
    """Share of the pool's nonland cards castable in its best two-color pair."""
    cards = [info.get(n) for n in names]
    cols = [set(c.colors) for c in cards if c is not None and "Land" not in c.types]
    if not cols:
        return float("nan")
    return max(sum(1 for cs in cols if cs <= set(pair)) for pair in PAIRS) / len(cols)


def color_fit(scorer, picks_all, rows: np.ndarray, card_names, info, n_drafts: int) -> tuple[float, float]:
    """Bot drafts its own pool along the packs each human saw; returns (bot fit, human fit).

    Approximate: later packs still reflect the human's picks, not the bot's."""
    meta = picks_all.meta.iloc[rows]
    drafts = sorted(meta["draft_id"].unique())[:n_drafts]
    by_draft = meta[meta["draft_id"].isin(drafts)].groupby("draft_id")
    bot_fits, human_fits = [], []
    for _, g in by_draft:
        g = g.sort_values(["pack_number", "pick_number"])
        pool: list[str] = []
        for r in g.index:
            pack = [card_names[i] for i, c in zip(picks_all.pack[r].indices, picks_all.pack[r].data) for _ in range(int(c))]
            ranked = scorer.rank(pack, pool)
            pool.append(ranked[0][0])
        last = g.index[-1]
        human = [card_names[i] for i, c in zip(picks_all.pool[last].indices, picks_all.pool[last].data) for _ in range(int(c))]
        if picks_all.pick[last] >= 0:
            human.append(card_names[picks_all.pick[last]])
        bot_fits.append(pool_fit(pool, info))
        human_fits.append(pool_fit(human, info))
    return float(np.nanmean(bot_fits)), float(np.nanmean(human_fits))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default="Cube_-_Powered")
    ap.add_argument("--event", default="PremierDraft")
    ap.add_argument("--synergy", type=float, nargs="+", default=[1.0])
    ap.add_argument("--deck-temps", type=float, nargs="+", default=[0.02, 0.05, 0.1],
                    help="Deck-aware scorer: pair softmax temperatures to try on validation")
    ap.add_argument("--deck-quantiles", type=float, nargs="+", default=[0.5, 0.7],
                    help="Deck-aware scorer: replacement-level quantiles to try on validation")
    ap.add_argument("--fit-drafts", type=int, default=300,
                    help="Drafts per split for the bot color-fit check (0 to skip)")
    args = ap.parse_args()

    t0 = time.time()
    raw = REPO_ROOT / "data" / "raw"
    models = REPO_ROOT / "data" / "models"
    fm, names = FactorizationMachine.load(models / f"fm.{args.set}.{args.event}.npz")
    lr, _ = FactorizationMachine.load(models / f"logreg.{args.set}.{args.event}.npz")
    card_names = names["cards"]

    games, _ = load_deck_data(raw, args.set, args.event)
    train_games = games.subset(np.flatnonzero(split_label(games.draft_id) == 0))
    lwr = card_logit_win_rates(train_games.cards, train_games.won)

    picks_all = load_picks(raw / f"draft_data_public.{args.set}.{args.event}.csv.gz", card_names,
                           keep_draft=lambda d: split_label(d) > 0, cache_tag="valtest")
    lab = split_label(picks_all.meta["draft_id"].to_numpy())
    pack_size = np.diff(picks_all.pack.indptr)
    print(f"Loaded models and {picks_all.n} val/test picks  [{time.time() - t0:.0f}s]")

    try:
        from mtgdraft.carddb import ArenaCardDB
        info = ArenaCardDB().by_name()
    except FileNotFoundError:
        info = None
        print("Arena card database not found: skipping the deck-aware scorer and color fit.")

    def deck(t, q):
        return DeckAwareScorer(fm, card_names, info, temperature=t, replacement_quantile=q)

    out, best_deck, best_wpa = {}, None, float("-inf")
    for split_name, split_id in (("val", 1), ("test", 2)):
        keep = np.flatnonzero((lab == split_id) & (pack_size > 1) & (picks_all.pick >= 0))
        from mtgdraft.model.drafts import PickData
        picks = PickData(picks_all.pack[keep], picks_all.pool[keep], picks_all.pick[keep],
                         picks_all.meta.iloc[keep].reset_index(drop=True))
        m = picks.meta
        strong = ((m["user_game_win_rate_bucket"] >= STRONG_WR) & (m["user_n_games_bucket"] >= STRONG_GAMES)).to_numpy()
        stage = np.where(m["pack_number"] == 0, np.where(m["pick_number"] <= 4, "P1 early", "P1 late"), "P2-P3")
        print(f"\n{split_name}: {picks.n} picks from {m['draft_id'].nunique()} drafts "
              f"({strong.mean():.0%} by strong drafters); stages P1 early = P1P2-P1P5, P1 late, P2-P3")
        res = []
        evaluate("card win rate", linear_scorer(lwr, card_names), picks, strong, stage, res)
        evaluate("logreg value", linear_scorer(lr.w_card, card_names), picks, strong, stage, res)
        evaluate("fm value only", PickScorer(fm, card_names, 0.0), picks, strong, stage, res)
        for a in args.synergy:
            evaluate(f"fm + synergy x{a:g}", PickScorer(fm, card_names, a), picks, strong, stage, res)
        if info is not None:
            # Tune on validation (by wins/agree); on test, report only the chosen setting.
            grid = [(t, q) for t in args.deck_temps for q in args.deck_quantiles] if split_name == "val" else [best_deck]
            for t, q in grid:
                r = evaluate(f"deck-aware t={t:g} q={q:g}", deck(t, q), picks, strong, stage, res)
                if split_name == "val" and r["wins_per_agree"] > best_wpa:
                    best_deck, best_wpa = (t, q), r["wins_per_agree"]
            if args.fit_drafts:
                rows = np.flatnonzero(lab == split_id)
                fm_fit, human_fit = color_fit(PickScorer(fm, card_names, 1.0), picks_all, rows, card_names, info, args.fit_drafts)
                dk_fit, _ = color_fit(deck(*best_deck), picks_all, rows, card_names, info, args.fit_drafts)
                print(f"  color fit (share of nonland cards castable in best 2-color pair, {args.fit_drafts} drafts): "
                      f"humans {human_fit:.0%} | fm + synergy {fm_fit:.0%} | deck-aware {dk_fit:.0%}", flush=True)
                res.append({"scorer": "color_fit", "human": human_fit, "fm_synergy": fm_fit,
                            "deck_aware": dk_fit, "deck_setting": best_deck, "n_drafts": args.fit_drafts})
        out[split_name] = res

    path = models / f"pick_eval.{args.set}.{args.event}.json"
    path.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nSaved {path.relative_to(REPO_ROOT)}  [{time.time() - t0:.0f}s total]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
