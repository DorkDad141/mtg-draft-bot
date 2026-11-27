"""Does deck structure (curve, creatures, lands) improve the deck model on held-out games?

Fits the deck model with its saved best settings twice, without and with the
structure features of mtgdraft/model/structure.py, and compares test log-loss
(paired, SE clustered by draft). Prints the learned curve: the win-rate effect of
each count, and the count where it peaks.

    python scripts/eval_structure.py --set Cube_-_Powered
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mtgdraft.carddb import ArenaCardDB  # noqa: E402
from mtgdraft.model.fm import FactorizationMachine, FMConfig  # noqa: E402
from mtgdraft.model.games import load_deck_data, split_by_draft  # noqa: E402
from mtgdraft.model.metrics import log_loss, paired_diff  # noqa: E402
from mtgdraft.model.structure import (GROUPS, SQUARE_SCALE, card_groups, structure_counts,  # noqa: E402
                                      structure_features)


def best_config(set_code, event):
    res = json.loads((REPO_ROOT / "data" / "models" / f"fm.{set_code}.{event}.results.json").read_text())
    val = res["val"]
    lr = min((v, k) for k, v in val.items() if k.startswith("logreg"))[1]
    fm = min((v, k) for k, v in val.items() if k.startswith("fm"))[1]
    l2w = float(lr.split("=")[1])
    k = int(fm.split("k=")[1].split()[0])
    l2v = float(fm.split("l2_v=")[1])
    return k, l2w, l2v


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default="Cube_-_Powered")
    ap.add_argument("--event", default="PremierDraft")
    args = ap.parse_args()
    t0 = time.time()
    data, source = load_deck_data(REPO_ROOT / "data" / "raw", args.set, args.event)
    info = ArenaCardDB().by_name()
    G = card_groups(data.card_names, info)
    counts = structure_counts(data.cards, G)
    S = sp.csr_matrix(structure_features(counts))
    tr, va, te = split_by_draft(data.draft_id)
    k, l2w, l2v = best_config(args.set, args.event)
    print(f"{args.set} ({source}): {data.n_games} rows; FM k={k} l2_w={l2w:g} l2_v={l2v:g}  [{time.time() - t0:.0f}s]")
    print("mean deck counts: " + ", ".join(f"{g} {m:.1f}" for g, m in zip(GROUPS, counts[tr].mean(0))))

    def run(with_struct):
        cov = sp.hstack([data.covariates, S], format="csr") if with_struct else data.covariates
        fm = FactorizationMachine(FMConfig(k=k, l2_w=l2w, l2_v=l2v)).fit(
            data.cards[tr], cov[tr], data.won[tr], val=(data.cards[va], cov[va], data.won[va]), verbose=False)
        return fm, fm.predict_proba(data.cards[te], cov[te])

    base_fm, p0 = run(False)
    print(f"  without structure: test log-loss {log_loss(data.won[te], p0):.5f}  [{time.time() - t0:.0f}s]", flush=True)
    fm, p1 = run(True)
    d, se = paired_diff(data.won[te], p1, p0, data.draft_id[te])
    print(f"  with structure:    test log-loss {log_loss(data.won[te], p1):.5f}  ({d:+.5f} +/- {se:.5f})  "
          f"[{time.time() - t0:.0f}s]")
    w = fm.w_cov[-len(GROUPS) * 2:].astype(np.float64)
    lin, sq = w[:len(GROUPS)], w[len(GROUPS):] / SQUARE_SCALE
    print("learned structure effect (logit per deck, relative to the mean deck), peak count:")
    mean = counts[tr].mean(0)
    for j, g in enumerate(GROUPS):
        f = lambda n: lin[j] * n + sq[j] * n * n  # noqa: E731
        peak = -lin[j] / (2 * sq[j]) if sq[j] < 0 else float("nan")
        lo, hi = max(0, mean[j] - 3), mean[j] + 3
        print(f"  {g:<10} {f(lo) - f(mean[j]):+.3f} at {lo:.0f}, {f(hi) - f(mean[j]):+.3f} at {hi:.0f}"
              + (f", peak at {peak:.1f}" if peak == peak else ", no peak (convex)"))
    out = REPO_ROOT / "data" / "models" / f"structure.{args.set}.{args.event}.json"
    out.write_text(json.dumps({"delta_logloss": d, "se": se, "w": w.tolist(), "groups": GROUPS,
                               "mean_counts": mean.tolist()}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
