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
from mtgdraft.model.games import load_games, split_label  # noqa: E402
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default="Cube_-_Powered")
    ap.add_argument("--event", default="PremierDraft")
    ap.add_argument("--synergy", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0])
    args = ap.parse_args()

    t0 = time.time()
    raw = REPO_ROOT / "data" / "raw"
    models = REPO_ROOT / "data" / "models"
    fm, names = FactorizationMachine.load(models / f"fm.{args.set}.{args.event}.npz")
    lr, _ = FactorizationMachine.load(models / f"logreg.{args.set}.{args.event}.npz")
    card_names = names["cards"]

    games = load_games(raw / f"game_data_public.{args.set}.{args.event}.csv.gz")
    train_games = games.subset(np.flatnonzero(split_label(games.draft_id) == 0))
    lwr = card_logit_win_rates(train_games.cards, train_games.won)

    picks_all = load_picks(raw / f"draft_data_public.{args.set}.{args.event}.csv.gz", card_names,
                           keep_draft=lambda d: split_label(d) > 0, cache_tag="valtest")
    lab = split_label(picks_all.meta["draft_id"].to_numpy())
    pack_size = np.diff(picks_all.pack.indptr)
    print(f"Loaded models and {picks_all.n} val/test picks  [{time.time() - t0:.0f}s]")

    out = {}
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
        out[split_name] = res

    path = models / f"pick_eval.{args.set}.{args.event}.json"
    path.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nSaved {path.relative_to(REPO_ROOT)}  [{time.time() - t0:.0f}s total]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
