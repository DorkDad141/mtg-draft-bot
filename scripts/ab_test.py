"""A/B test two pick scorers in simulated 8-player pods, judged by an independent model.

Pods: each pod deals 24 real opening packs (pick 1 of a pack in held-out 17Lands
drafts, so rarity and collation are real) to 8 seats that pass left, right, left. Seats
alternate scorer A and scorer B (ABABABAB), then the same pod is replayed with the
seats swapped (BABABABA), so both scorers see the same packs from the same seats.

Decks: every final pool goes through the same neutral deck builder. For each candidate
deck (the 70 color pairs / triples / splashes), it takes the spells the mana base can
cast, counting each spell by its exact castability probability p (hypergeometric, see
mtgdraft/model/mana.py): it adds spells greedily by p x (judge value + synergy) + (1 - p) x dead up to
23, adds the pool's lands in those colors and basics by pip demand to 17, and keeps the
deck the judge rates highest, where each spell's uncastable share (1 - p) is valued as a
dead card (5th-percentile card value). The same adjustment is applied to the judged win
rate. (The judge alone would not avoid uncastable cards: real decks, its training data,
never contain them.)

Judge: a win-rate model (same kind and settings as the scorers' model) trained only on
the held-out drafts (val + test), which the scorers' model never saw, so neither
scorer is graded by the model it was built on. The scorers' own model is reported too,
as a reference ("self-judged"). Independent checks: how castable the deck's spells
are, playables, and deck shapes.

Two deck builds are reported: "free" (the colors the judge likes best) and
"human-style" (colors from the lane-choice model fitted on what humans built, then the
best 23 in them). Result: per pod, mean judged win rate of A's decks minus B's (both
seatings averaged), with a standard error over pods.

    python scripts/ab_test.py --set TRK --pods 100
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.sparse as sp

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mtgdraft.carddb import ArenaCardDB  # noqa: E402
from mtgdraft.model.drafts import load_picks  # noqa: E402
from mtgdraft.model.fm import FactorizationMachine, FMConfig  # noqa: E402
from mtgdraft.model.games import load_deck_data, split_label  # noqa: E402
from mtgdraft.model.mana_scorer import ManaAwareScorer  # noqa: E402
from mtgdraft.model.scorer import BASIC_LANDS, load_scorer, shrink_by_evidence  # noqa: E402
from scripts.eval_structure import best_config  # noqa: E402

MODELS = REPO_ROOT / "data" / "models"
BASIC_OF = {"W": "Plains", "U": "Island", "B": "Swamp", "R": "Mountain", "G": "Forest"}


# ---- judge -------------------------------------------------------------------------
def train_judge(set_code, event, card_names):
    """FM trained on held-out drafts only (val + test), with the set's best settings."""
    path = MODELS / f"judge.{set_code}.{event}.npz"
    if path.exists():
        fm, names = FactorizationMachine.load(path)
        if names["cards"] == card_names:
            return fm
    data, _ = load_deck_data(REPO_ROOT / "data" / "raw", set_code, event)
    assert data.card_names == card_names, "judge and scorer card lists differ"
    lab = split_label(data.draft_id)
    held = np.flatnonzero(lab > 0)
    inner = split_label(data.draft_id[held], val_frac=0.15, test_frac=0.0, salt="judge")
    fit_i, stop_i = held[inner == 0], held[inner == 1]
    k, l2w, l2v = best_config(set_code, event)
    fm = FactorizationMachine(FMConfig(k=k, l2_w=l2w, l2_v=l2v)).fit(
        data.cards[fit_i], data.covariates[fit_i], data.won[fit_i],
        val=(data.cards[stop_i], data.covariates[stop_i], data.won[stop_i]), verbose=False)
    fm.save(path, data.card_names, data.covariate_names, set=set_code, event=event,
            note="judge: trained on val+test drafts only")
    print(f"Trained judge on {len(fit_i)} held-out rows ({len(stop_i)} for early stopping)")
    return fm


# ---- deck building -----------------------------------------------------------------
class DeckBuilder:
    def __init__(self, judge: FactorizationMachine, card_names, info):
        self.judge = judge
        lm_path = MODELS / "lane_model.json"
        lm = json.loads(lm_path.read_text()) if lm_path.exists() else {}
        # Mana bases, castability, and the human-fitted lane-choice model (for "human-style" decks).
        self.ms = ManaAwareScorer(judge, card_names, info, lane_beta=lm.get("beta"),
                                  slack=lm.get("slack", 2), splash_basics=lm.get("splash_basics", 2))
        self.names = card_names
        self.n_cov = judge.w_cov.shape[0]
        self.V = judge.V.astype(np.float64) if judge.config.k else np.zeros((len(card_names), 0))
        self.w = judge.w_card.astype(np.float64)
        self.basic_idx = {c: card_names.index(b) for c, b in BASIC_OF.items() if b in card_names}
        # A spell that can't be cast is a dead card: worth a near-useless card (5th percentile).
        self.dead = float(np.quantile(self.w[self.ms._spell], 0.05))

    def build(self, pool: list[str]):
        """Best 40-card deck for a pool: 23 spells + 17 lands. Each candidate deck is scored by
        the judge, then each spell's uncastable share (1 - p) is valued as a dead card."""
        ms = self.ms
        idx = np.array([ms._idx[n] for n in pool if n in ms._idx], dtype=int)
        spells = idx[ms._spell[idx]]
        nb_lands = idx[ms._is_land[idx] & ~ms._is_basic[idx]]
        src = ms._sources(idx)
        L = len(ms.lanes_def)
        p_all = ms.mana.castable(spells, src, ms.slack)                # (L, m)
        X = sp.lil_matrix((L, len(self.names)), dtype=np.float32)
        info = []
        for li in range(L):
            p = p_all[li]
            chosen, vsum = [], np.zeros(self.V.shape[1])
            avail = list(range(len(spells)))
            while avail and len(chosen) < 23:
                cand = spells[avail]
                gain = p[avail] * (self.w[cand] + self.V[cand] @ vsum) + (1 - p[avail]) * self.dead
                j = int(np.argmax(gain))
                k = avail.pop(j)
                chosen.append(k)
                vsum += self.V[spells[k]]
            lane_cols = ms._lane_mask[li]
            # Same land rule as the scorer: a nonbasic land is played if it makes 2+ of the
            # deck's colors (a single-color nonbasic is only a worse basic).
            lands = [i for i in nb_lands if (ms.mana.produces[i] & lane_cols).sum() >= 2][:8]
            n_basic = 17 - len(lands)
            demand = np.zeros(5)
            for j in chosen:
                demand += ms.mana.single[spells[j]] * p[j]
            demand = np.where(lane_cols, demand + 0.5, 0.0)
            alloc = np.floor(n_basic * demand / demand.sum()).astype(int)
            while alloc.sum() < n_basic:
                alloc[int(np.argmax(np.where(lane_cols, demand / (alloc + 1), -1)))] += 1
            for j in chosen:
                X[li, spells[j]] += 1.0
            for i in lands:
                X[li, i] += 1.0
            for x, c in enumerate("WUBRG"):
                if alloc[x] and c in self.basic_idx:
                    X[li, self.basic_idx[c]] += float(alloc[x])
            pc = p[chosen] if chosen else np.zeros(1)
            penalty = float(sum((1 - p[j]) * (self.w[spells[j]] - self.dead) for j in chosen))
            info.append((li, float(pc.mean()), int((pc >= 0.5).sum()), penalty))
        X = X.tocsr()
        logit = self.judge.logit(X, sp.csr_matrix((L, self.n_cov), dtype=np.float32))
        logit = logit - np.array([x[3] for x in info])
        out = {}
        human = int(np.argmax(ms.lane_features(idx) @ np.asarray(ms.lane_beta))) if ms.lane_beta is not None             else int(np.argmax(logit))
        for mode, best in (("free", int(np.argmax(logit))), ("human", human)):
            li, mean_p, n_castable, penalty = info[best]
            out[mode] = {"lane": ms.lanes_def[li], "deck": X[best], "mean_p": mean_p, "castable": n_castable,
                         "penalty": penalty, "logit": float(logit[best])}
        return out


def rate_deck(builder: "DeckBuilder", spells: list[str], castable: list[float], lands: list[str],
              basics: dict) -> float:
    """Judge logit of a given 40-card deck (e.g. one the scorer built), castability-adjusted
    the same way as DeckBuilder.build: each spell's uncastable share is a dead card."""
    idx = {n: i for i, n in enumerate(builder.names)}
    X = sp.lil_matrix((1, len(builder.names)), dtype=np.float32)
    penalty = 0.0
    for n, p in zip(spells, castable):
        if n in idx:
            X[0, idx[n]] += 1.0
            penalty += (1 - p) * (builder.w[idx[n]] - builder.dead)
    for n in lands:
        if n in idx:
            X[0, idx[n]] += 1.0
    for n, c in basics.items():
        if n in idx:
            X[0, idx[n]] += float(c)
    logit = float(builder.judge.logit(X.tocsr(), sp.csr_matrix((1, builder.n_cov), dtype=np.float32))[0])
    return logit - penalty


def deck_logit(fm, X, n_cov):
    return float(fm.logit(X, sp.csr_matrix((1, n_cov), dtype=np.float32))[0])


# ---- pods --------------------------------------------------------------------------
def opening_packs(set_code, event, card_names):
    picks = load_picks(REPO_ROOT / "data" / "raw" / f"draft_data_public.{set_code}.{event}.csv.gz", card_names,
                       keep_draft=lambda d: split_label(d) > 0, cache_tag="valtest")
    rows = np.flatnonzero((picks.meta["pick_number"] == 0).to_numpy())
    packs = []
    for r in rows:
        row = picks.pack[r]
        packs.append([card_names[i] for i, c in zip(row.indices, row.data) for _ in range(int(c))])
    return [p for p in packs if len(p) >= 10]


def run_pod(packs24, scorers):
    """scorers: 8 scorer objects by seat. Returns 8 final pools."""
    pools = [[] for _ in range(8)]
    for rnd in range(3):
        current = [list(packs24[rnd * 8 + s]) for s in range(8)]
        step = 1 if rnd != 1 else -1
        while any(current):
            for s in range(8):
                if current[s]:
                    pick = scorers[s].rank(current[s], pools[s])[0][0]
                    current[s].remove(pick)
                    pools[s].append(pick)
            current = [current[(s - step) % 8] for s in range(8)]
    return pools


def shape(lane):
    main, spl = lane
    return f"{len(main)}" + ("+splash" if spl else "")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default="TRK")
    ap.add_argument("--event", default="PremierDraft")
    ap.add_argument("--pods", type=int, default=100)
    ap.add_argument("--a", default="mana", help="Scorer A (load_scorer kind)")
    ap.add_argument("--b", default="deck", help="Scorer B")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    info = ArenaCardDB().by_name()
    A = load_scorer(args.set, args.event, card_info=info, kind=args.a)
    B = load_scorer(args.set, args.event, card_info=info, kind=args.b)
    card_names = A.card_names
    own, _ = FactorizationMachine.load(MODELS / f"fm.{args.set}.{args.event}.npz")
    shrink_by_evidence(own, _.get("meta", {}).get("card_decks"))
    judge = train_judge(args.set, args.event, card_names)
    builder = DeckBuilder(judge, card_names, info)
    packs = opening_packs(args.set, args.event, card_names)
    print(f"{args.set}: A = {args.a}, B = {args.b}; {len(packs)} real opening packs; {args.pods} pods x 2 seatings  "
          f"[{time.time() - t0:.0f}s]", flush=True)

    rng = np.random.default_rng(args.seed)
    n_cov = judge.w_cov.shape[0]
    per_pod = []
    stats = {"A": [], "B": []}
    for pod in range(args.pods):
        chosen = rng.choice(len(packs), 24, replace=False)
        p24 = [packs[i] for i in chosen]
        diff_j, diff_s, diff_h = [], [], []
        for seating in (0, 1):
            arms = ["A" if (s + seating) % 2 == 0 else "B" for s in range(8)]
            pools = run_pod(p24, [A if a == "A" else B for a in arms])
            res = {"A": [], "B": []}
            for arm, pool in zip(arms, pools):
                built = builder.build(pool)
                d, h = built["free"], built["human"]
                sl = deck_logit(own, d["deck"], n_cov) - d["penalty"]
                spells = [n for n in pool if n in info and "Land" not in info[n].types and n not in BASIC_LANDS]
                sig = lambda z: 1 / (1 + np.exp(-z))  # noqa: E731
                row = {"judge": sig(d["logit"]), "self": sig(sl), "human": sig(h["logit"]),
                       "shape": shape(d["lane"]), "human_shape": shape(h["lane"]), "mean_p": d["mean_p"],
                       "human_mean_p": h["mean_p"], "castable": d["castable"], "lands_picked": len(pool) - len(spells)}
                res[arm].append(row)
                stats[arm].append(row)
            diff_j.append(np.mean([r["judge"] for r in res["A"]]) - np.mean([r["judge"] for r in res["B"]]))
            diff_s.append(np.mean([r["self"] for r in res["A"]]) - np.mean([r["self"] for r in res["B"]]))
            diff_h.append(np.mean([r["human"] for r in res["A"]]) - np.mean([r["human"] for r in res["B"]]))
        per_pod.append((np.mean(diff_j), np.mean(diff_s), np.mean(diff_h)))
        if (pod + 1) % 10 == 0:
            d = np.array(per_pod)
            sd = 100 * d.std(0, ddof=1) / np.sqrt(len(d))
            print(f"  {pod + 1} pods: judged A - B: free build {100 * d[:, 0].mean():+.2f} (+/- {sd[0]:.2f}), "
                  f"human-style {100 * d[:, 2].mean():+.2f} (+/- {sd[2]:.2f}) pts  [{time.time() - t0:.0f}s]", flush=True)

    d = np.array(per_pod)
    se = d.std(0, ddof=1) / np.sqrt(len(d))
    out = {"set": args.set, "a": args.a, "b": args.b, "pods": args.pods,
           "judge_diff_pts": 100 * d[:, 0].mean(), "judge_se_pts": 100 * se[0],
           "self_diff_pts": 100 * d[:, 1].mean(), "self_se_pts": 100 * se[1],
           "human_diff_pts": 100 * d[:, 2].mean(), "human_se_pts": 100 * se[2]}
    print(f"\n{args.set}, {args.pods} pods ({8 * args.pods} decks per scorer):")
    print(f"  judged win rate, free deck build (independent judge):       A - B {out['judge_diff_pts']:+.2f} pts "
          f"+/- {out['judge_se_pts']:.2f}")
    print(f"  judged win rate, human-style deck build (independent judge): A - B {out['human_diff_pts']:+.2f} pts "
          f"+/- {out['human_se_pts']:.2f}")
    print(f"  judged win rate, free deck build (scorers' own model):       A - B {out['self_diff_pts']:+.2f} pts "
          f"+/- {out['self_se_pts']:.2f}")
    for arm, name in (("A", args.a), ("B", args.b)):
        rows = stats[arm]
        sh = Counter(r["shape"] for r in rows)
        hsh = Counter(r["human_shape"] for r in rows)
        out[arm] = {"judge_wr": float(np.mean([r["judge"] for r in rows])),
                    "mean_castability": float(np.mean([r["mean_p"] for r in rows])),
                    "castable_spells": float(np.mean([r["castable"] for r in rows])),
                    "lands_picked": float(np.mean([r["lands_picked"] for r in rows])),
                    "shapes": {k: v / len(rows) for k, v in sh.items()},
                    "human_style_shapes": {k: v / len(rows) for k, v in hsh.items()},
                    "human_style_castability": float(np.mean([r["human_mean_p"] for r in rows]))}
        o = out[arm]
        print(f"  {arm} {name:<5}: judged WR {100 * o['judge_wr']:.1f}%, deck spells' mean castability "
              f"{o['mean_castability']:.2f}, castable spells {o['castable_spells']:.1f}/23, lands picked "
              f"{o['lands_picked']:.1f}\n          free decks " + " ".join(f"{k} {v:.0%}" for k, v in sorted(o["shapes"].items()))
              + " | human-style decks " + " ".join(f"{k} {v:.0%}" for k, v in sorted(o["human_style_shapes"].items()))
              + f" (castability {o['human_style_castability']:.2f})")
    path = MODELS / f"ab_test.{args.set}.{args.event}.{args.a}_vs_{args.b}.json"
    path.write_text(json.dumps(out, indent=2, default=float))
    print(f"Saved {path.relative_to(REPO_ROOT)}  [{time.time() - t0:.0f}s]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
