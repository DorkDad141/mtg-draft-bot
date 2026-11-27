"""Fit the lane-choice model: which deck (colors + splash) a pool becomes.

A conditional logit over the scorer's lanes (pairs, triples, pair + splash, triple +
splash): P(lane | pool) is proportional to exp(beta . features(lane, pool)), with the
card-agnostic features in mana_scorer.LANE_FEATURES (castable deck value, depth of
playables, thinnest main color, triple / splash flags, number of splash cards).

It is trained on what humans actually built (17Lands game data records each deck's
main and splash colors), pooling every set given, so one small set of weights (6
numbers) is shared by all sets, including sets without game data such as TRK. The
mana model's knobs (slack, splash basics) are chosen by validation likelihood.
Reported on held-out drafts per set, next to the "pair with the most cards" baseline.

    python scripts/fit_lane_model.py --sets Cube_-_Powered SOS
Saves data/models/lane_model.json, which load_scorer uses for every set.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
import zlib
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mtgdraft.carddb import ArenaCardDB  # noqa: E402
from mtgdraft.model.drafts import load_picks  # noqa: E402
from mtgdraft.model.fm import FactorizationMachine  # noqa: E402
from mtgdraft.model.games import split_label  # noqa: E402
from mtgdraft.model.mana_scorer import LANE_FEATURES, ManaAwareScorer  # noqa: E402
from mtgdraft.model.scorer import shrink_by_evidence  # noqa: E402
from scripts.eval_mana import final_pools, human_decks, shape  # noqa: E402

LAMBDA = 1e-3


def subsample(ids: np.ndarray, every: int) -> np.ndarray:
    return np.array([zlib.crc32(str(d).encode()) % every == 0 for d in ids])


def load_set(set_code, event, info, train_every: int, n_eval: int):
    raw, models = REPO_ROOT / "data" / "raw", REPO_ROOT / "data" / "models"
    fm, names = FactorizationMachine.load(models / f"fm.{set_code}.{event}.npz")
    shrink_by_evidence(fm, names.get("meta", {}).get("card_decks"))
    cards = names["cards"]
    path = raw / f"draft_data_public.{set_code}.{event}.csv.gz"
    out = {"fm": fm, "cards": cards}
    tr = load_picks(path, cards, keep_draft=lambda d: (split_label(d) == 0) & subsample(d, train_every),
                    cache_tag=f"lanefit{train_every}")
    vt = load_picks(path, cards, keep_draft=lambda d: split_label(d) > 0, cache_tag="valtest")
    from mtgdraft.model.drafts import PickData
    lab = split_label(vt.meta["draft_id"].to_numpy())
    pools = {"train": final_pools(tr, cards)}
    for name, sid in (("val", 1), ("test", 2)):
        ids = set(sorted(set(vt.meta["draft_id"][lab == sid]))[:n_eval])
        keep = np.flatnonzero(vt.meta["draft_id"].isin(ids).to_numpy())
        sub = PickData(vt.pack[keep], vt.pool[keep], vt.pick[keep], vt.meta.iloc[keep].reset_index(drop=True))
        pools[name] = final_pools(sub, cards)
    all_ids = set().union(*[set(p) for p in pools.values()])
    out["humans"] = human_decks(raw, set_code, event, all_ids)
    out["pools"] = pools
    return out


def build(data, info, slack, splash_basics):
    sc = ManaAwareScorer(data["fm"], data["cards"], info, slack=slack, splash_basics=splash_basics)
    lane_of = {lane: i for i, lane in enumerate(sc.lanes_def)}
    sets = {}
    for split, pools in data["pools"].items():
        X, y, keys = [], [], []
        for d, pool in pools.items():
            if d not in data["humans"].index:
                continue
            m, s = data["humans"].loc[d, "main"], data["humans"].loc[d, "splash"]
            li = lane_of.get((m, s)) if len(s) <= 1 else None
            idx = np.array([sc._idx[n] for n in pool if n in sc._idx], dtype=int)
            X.append(sc.lane_features(idx))
            y.append(-1 if li is None else li)
            keys.append((d, m, s, idx))
        sets[split] = (np.array(X), np.array(y), keys)
    return sc, sets


def nll(beta, X, y):
    z = X @ beta                                   # (N, L)
    z = z - z.max(1, keepdims=True)
    lse = np.log(np.exp(z).sum(1))
    ok = y >= 0
    ll = z[ok, y[ok]] - lse[ok]
    p = np.exp(z - lse[:, None])
    grad = -(X[ok, y[ok]] - np.einsum("nl,nlf->nf", p[ok], X[ok])).sum(0)
    n = ok.sum()
    return -ll.sum() / n + LAMBDA * beta @ beta, grad / n + 2 * LAMBDA * beta


def fit(Xs, ys):
    X, y = np.concatenate(Xs), np.concatenate(ys)
    r = minimize(nll, np.zeros(X.shape[2]), args=(X, y), jac=True, method="L-BFGS-B")
    return r.x


def report(sc, beta, X, y, keys, cards, info):
    pred = (X @ beta).argmax(1)
    main_ok = shape_ok = base_ok = 0
    pairs = ["".join(p) for p in combinations("WUBRG", 2)]
    bot_sh, hum_sh = {}, {}
    for (d, m, s, idx), li in zip(keys, pred):
        bm, bs = sc.lanes_def[li]
        main_ok += bm == m
        shape_ok += shape(bm, bs) == shape(m, s)
        spells = [i for i in idx if sc._spell[i]]
        best = max(pairs, key=lambda p: sum(1 for i in spells if sc.mana.colors[i] and sc.mana.colors[i] <= set(p)))
        base_ok += best == m
        bot_sh[shape(bm, bs)] = bot_sh.get(shape(bm, bs), 0) + 1
        hum_sh[shape(m, s)] = hum_sh.get(shape(m, s), 0) + 1
    n = len(keys)
    ll = -nll(beta, X, y)[0]
    fmt = lambda c: " ".join(f"{k} {v / n:.0%}" for k, v in sorted(c.items(), key=lambda kv: -kv[1])[:5])
    return (f"main colors {main_ok / n:.1%} (count baseline {base_ok / n:.1%}), shape {shape_ok / n:.1%}, "
            f"log-lik {ll:.3f}\n      bot: {fmt(bot_sh)}\n      human: {fmt(hum_sh)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", nargs="+", default=["Cube_-_Powered", "SOS"])
    ap.add_argument("--event", default="PremierDraft")
    ap.add_argument("--train-every", type=int, default=8, help="Use 1 in N training drafts")
    ap.add_argument("--eval-drafts", type=int, default=1500)
    ap.add_argument("--slack", type=float, nargs="+", default=[1, 2])
    ap.add_argument("--splash-basics", type=int, nargs="+", default=[2, 3])
    args = ap.parse_args()

    t0 = time.time()
    info = ArenaCardDB().by_name()
    data = {s: load_set(s, args.event, info, args.train_every, args.eval_drafts) for s in args.sets}
    print(f"Loaded {', '.join(f'{s}: {len(d['pools']['train'])} train drafts' for s, d in data.items())}  "
          f"[{time.time() - t0:.0f}s]", flush=True)
    best = None
    for slack, sb in itertools.product(args.slack, args.splash_basics):
        built = {s: build(d, info, slack, sb) for s, d in data.items()}
        n_min = min(len(b[1]["train"][1]) for b in built.values())
        beta = fit([b[1]["train"][0][:n_min] for b in built.values()], [b[1]["train"][1][:n_min] for b in built.values()])
        val_ll = np.mean([-nll(beta, *b[1]["val"][:2])[0] for b in built.values()])
        print(f"\nslack {slack:g}, splash basics {sb}: val log-lik {val_ll:.4f}  beta "
              + ", ".join(f"{f} {b:+.2f}" for f, b in zip(LANE_FEATURES, beta)) + f"  [{time.time() - t0:.0f}s]", flush=True)
        for s, (sc, sets) in built.items():
            print(f"  {s} val: " + report(sc, beta, *sets["val"], data[s]["cards"], info), flush=True)
        if best is None or val_ll > best[0]:
            best = (val_ll, slack, sb, beta, built)
    _, slack, sb, beta, built = best
    print(f"\nChosen: slack {slack:g}, splash basics {sb}. Held-out test:")
    metrics = {}
    for s, (sc, sets) in built.items():
        txt = report(sc, beta, *sets["test"], data[s]["cards"], info)
        print(f"  {s} test: {txt}")
        metrics[s] = txt
    out = {"features": LANE_FEATURES, "beta": [float(b) for b in beta], "slack": slack, "splash_basics": sb,
           "trained_on": args.sets, "test": metrics, "fitted": time.strftime("%Y-%m-%d")}
    path = REPO_ROOT / "data" / "models" / "lane_model.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"Saved {path.relative_to(REPO_ROOT)}  [{time.time() - t0:.0f}s]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
