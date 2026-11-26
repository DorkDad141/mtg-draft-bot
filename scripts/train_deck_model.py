"""Train and evaluate deck win-rate models on 17Lands game data.

Compares, on held-out drafts:
  prior        constant win rate
  controls     logistic regression on controls only (skill, rank, play/draw, mulligans)
  card_wr      controls + average smoothed per-card deck win rate (the "17Lands GIH WR" idea)
  logreg       controls + one weight per card (no synergy)
  fm_k{K}      controls + card weights + pairwise synergy (factorization machine)

The best FM (by validation log-loss) is saved to data/models/.

Example:
    python scripts/train_deck_model.py --set Cube_-_Powered --event PremierDraft
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mtgdraft.model.baselines import card_logit_win_rates  # noqa: E402
from mtgdraft.model.fm import FactorizationMachine, FMConfig  # noqa: E402
from mtgdraft.model.games import load_games, split_by_draft  # noqa: E402
from mtgdraft.model.metrics import auc, log_loss, paired_diff  # noqa: E402


def card_wr_feature(train, *others, prior_games: float = 200.0):
    """Count-weighted mean logit of each card's smoothed main-deck win rate."""
    lwr = card_logit_win_rates(train.cards, train.won, prior_games)

    def feat(d):
        x = d.cards
        tot = np.asarray(x.sum(1)).ravel()
        return sp.csr_matrix(((x @ lwr) / np.maximum(tot, 1))[:, None].astype(np.float32))

    return [feat(train)] + [feat(o) for o in others]


def evaluate(name, p_test, test, results, ref=None):
    r = {"model": name, "test_logloss": log_loss(test.won, p_test), "test_auc": auc(test.won, p_test)}
    if ref is not None:
        r["vs_logreg"], r["vs_logreg_se"] = paired_diff(test.won, p_test, ref, test.draft_id)
    results.append(r)
    extra = f"  vs logreg {r['vs_logreg']:+.5f} +/- {r['vs_logreg_se']:.5f}" if ref is not None else ""
    print(f"  {name:<22} log-loss {r['test_logloss']:.5f}  AUC {r['test_auc']:.4f}{extra}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default="Cube_-_Powered", help="Set code as in the file name")
    ap.add_argument("--event", default="PremierDraft")
    ap.add_argument("--ks", type=int, nargs="+", default=[4, 8, 16])
    ap.add_argument("--l2-v", type=float, nargs="+", default=[3e-4, 1e-3, 3e-3])
    ap.add_argument("--l2-w", type=float, nargs="+", default=[1e-5, 1e-4, 1e-3])
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    raw = REPO_ROOT / "data" / "raw" / f"game_data_public.{args.set}.{args.event}.csv.gz"
    if not raw.exists():
        sys.exit(f"{raw} not found. Run scripts/download_17lands.py first.")
    t0 = time.time()
    data = load_games(raw)
    tr_i, va_i, te_i = split_by_draft(data.draft_id)
    train, val, test = data.subset(tr_i), data.subset(va_i), data.subset(te_i)
    print(f"{data.n_games} games, {len(data.card_names)} cards; train/val/test games "
          f"{train.n_games}/{val.n_games}/{test.n_games} (split by draft)  [{time.time() - t0:.1f}s]")
    verbose = not args.quiet

    def fit(cfg, tr_cards, va_cards):
        return FactorizationMachine(cfg).fit(tr_cards, train.covariates, train.won,
                                             val=(va_cards, val.covariates, val.won), verbose=verbose)

    results, val_scores = [], {}

    # prior
    evaluate("prior", np.full(test.n_games, train.won.mean()), test, results)

    # controls only
    empty = lambda d: sp.csr_matrix((d.n_games, 0), dtype=np.float32)  # noqa: E731
    print("controls only")
    m = fit(FMConfig(k=0, l2_w=1e-5), empty(train), empty(val))
    evaluate("controls", m.predict_proba(empty(test), test.covariates), test, results)

    # per-card win rate average
    print("card win-rate average")
    ftr, fva, fte = card_wr_feature(train, val, test)
    m = fit(FMConfig(k=0, l2_w=1e-5), ftr, fva)
    evaluate("card_wr", m.predict_proba(fte, test.covariates), test, results)

    # logistic regression (tune l2_w on val)
    best_lr = None
    for l2w in args.l2_w:
        print(f"logreg l2_w={l2w:g}")
        m = fit(FMConfig(k=0, l2_w=l2w), train.cards, val.cards)
        v = log_loss(val.won, m.predict_proba(val.cards, val.covariates))
        val_scores[f"logreg l2_w={l2w:g}"] = v
        if best_lr is None or v < best_lr[0]:
            best_lr = (v, l2w, m)
    lr_model = best_lr[2]
    p_lr = lr_model.predict_proba(test.cards, test.covariates)
    evaluate(f"logreg (l2_w={best_lr[1]:g})", p_lr, test, results, ref=p_lr)

    # factorization machines (tune k, l2_v on val; l2_w from logreg)
    best_fm = None
    for k, l2v in itertools.product(args.ks, args.l2_v):
        print(f"fm k={k} l2_v={l2v:g}")
        m = fit(FMConfig(k=k, l2_w=best_lr[1], l2_v=l2v), train.cards, val.cards)
        v = log_loss(val.won, m.predict_proba(val.cards, val.covariates))
        val_scores[f"fm k={k} l2_v={l2v:g}"] = v
        print(f"    -> val log-loss {v:.5f}")
        if best_fm is None or v < best_fm[0]:
            best_fm = (v, k, l2v, m)
    fm = best_fm[3]
    p_fm = fm.predict_proba(test.cards, test.covariates)
    evaluate(f"fm (k={best_fm[1]}, l2_v={best_fm[2]:g})", p_fm, test, results, ref=p_lr)

    print("\nValidation log-loss by setting:")
    for k, v in val_scores.items():
        print(f"  {k:<28} {v:.5f}")

    out = REPO_ROOT / "data" / "models" / f"fm.{args.set}.{args.event}.npz"
    fm.save(out, data.card_names, data.covariate_names, set=args.set, event=args.event,
            test_results=results, trained_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
    lr_model.save(out.with_name(f"logreg.{args.set}.{args.event}.npz"), data.card_names,
                  data.covariate_names, set=args.set, event=args.event)
    (out.with_suffix(".results.json")).write_text(json.dumps(
        {"results": results, "val": val_scores}, indent=2))
    print(f"\nSaved {out.relative_to(REPO_ROOT)}  [{time.time() - t0:.0f}s total]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
