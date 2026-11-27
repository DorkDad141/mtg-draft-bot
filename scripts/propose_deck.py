"""Propose a 40-card deck from a recorded draft's pool, rate it, and compare it with human decks.

Uses the same pieces as the A/B tests (scripts/ab_test.py), so the numbers are
comparable with scripts/compare_decks.py:
  * deck builder: for every color pair / triple / splash, the 23 spells with the best
    castability-weighted value and synergy, the pool's lands in those colors, and basics
    by pip demand to 17 lands; the best-rated build is proposed. A second build uses the
    colors humans would most likely choose for this pool ("human-style").
  * rating: the independent judge (a win-rate model trained only on held-out drafts),
    with each spell's uncastable share valued as a dead card. Neutral controls (no skill
    or rank), so it rates the deck, not the player.
  * comparison: human pools from 17Lands drafts the judge never saw, built and rated the
    same way (scripts/compare_decks.py, which this reads): the average deck, the average
    deck with a winning match record, 7-win decks, and your deck's percentile.

    python scripts/propose_deck.py                 # newest recorded draft
    python scripts/propose_deck.py --draft QuickDraft_TRK
Writes captures/deck_<draft>.json, which draft_viewer.py shows as a final "Your deck" slide.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mtgdraft.carddb import ArenaCardDB  # noqa: E402
from mtgdraft.model.mana_scorer import lane_name  # noqa: E402
from mtgdraft.model.scorer import load_scorer  # noqa: E402
from scripts.ab_test import MODELS, DeckBuilder, train_judge  # noqa: E402
from scripts.draft_viewer import CAPTURES, load_sessions  # noqa: E402

CNAME = {"W": "White", "U": "Blue", "B": "Black", "R": "Red", "G": "Green"}


def describe(build, builder, pool_idx, info):
    """Card list of a built deck with each spell's castability in that deck's mana base."""
    ms = builder.ms
    li = ms.lanes_def.index(build["lane"])
    src = ms._sources(pool_idx)[li:li + 1]
    row = build["deck"]
    cards, lands = [], []
    for i, c in zip(row.indices, row.data):
        nm = builder.names[i]
        ci = info.get(nm)
        entry = {"name": nm, "count": int(round(c)), "mv": ci.mana_value if ci else None,
                 "colors": ci.colors if ci else "", "cost": ci.mana_cost if ci else "",
                 "type": " ".join(ci.types) if ci else ""}
        if ms._is_land[i]:
            lands.append(entry)
        else:
            entry["castable"] = round(float(ms.mana.castable(np.array([i]), src, ms.slack)[0, 0]), 2)
            cards.append(entry)
    cards.sort(key=lambda e: (e["mv"] if e["mv"] is not None else 99, e["name"]))
    lands.sort(key=lambda e: (e["name"] not in ("Plains", "Island", "Swamp", "Mountain", "Forest"), e["name"]))
    n_creatures = sum(e["count"] for e in cards if "Creature" in e["type"])
    curve = {}
    for e in cards:
        k = "6+" if (e["mv"] or 0) >= 6 else str(max(e["mv"] or 0, 1) if e["mv"] else 0)
        curve[k] = curve.get(k, 0) + e["count"]
    return {"lane": lane_name(*build["lane"]), "spells": cards, "lands": lands,
            "creatures": n_creatures, "curve": curve,
            "win_rate": float(1 / (1 + np.exp(-build["logit"]))), "mean_castability": round(build["mean_p"], 3)}


def compare(win_rate, humans, key):
    vals = np.array([h[key] for h in humans])
    win = np.array([h[key] for h in humans if h["wins"] > h["losses"]])
    trophy = np.array([h[key] for h in humans if h["wins"] >= 7])
    return {"avg": float(vals.mean()), "avg_winning": float(win.mean()) if len(win) else None,
            "avg_7_wins": float(trophy.mean()) if len(trophy) else None,
            "percentile": float((vals < win_rate).mean() * 100),
            "percentile_vs_winning": float((win < win_rate).mean() * 100) if len(win) else None,
            "n": int(len(vals)), "n_winning": int(len(win)), "n_7_wins": int(len(trophy))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--draft", help="Part of the draft id (default: the most recent recorded draft)")
    ap.add_argument("--event", default="PremierDraft", help="Model/comparison data to use")
    args = ap.parse_args()

    drafts = load_sessions(args.draft)
    if not drafts:
        sys.exit("No recorded drafts found in captures/.")
    did, d = max(drafts.items(), key=lambda kv: kv[1]["last_t"])
    if d["complete"]:
        pool = list(d["complete"]["names"])
    else:
        pool = [p["name"] for _, p in sorted(d["picks"].items())]
    set_code = next(iter(d["packs"].values()))["set"]
    info = ArenaCardDB().by_name()
    scorer = load_scorer(set_code, args.event, card_info=info)
    if scorer is None:
        sys.exit(f"No model for {set_code}.")
    names = scorer.card_names
    judge = train_judge(set_code, args.event, names)
    builder = DeckBuilder(judge, names, info)
    pool_idx = np.array([builder.ms._idx[n] for n in pool if n in builder.ms._idx], dtype=int)
    built = builder.build(pool)
    best = describe(built["free"], builder, pool_idx, info)
    alt = describe(built["human"], builder, pool_idx, info)

    cmp_path = MODELS / f"compare_decks.{set_code}.{args.event}.json"
    humans = json.loads(cmp_path.read_text()).get("human_decks") if cmp_path.exists() else None
    if humans:
        best["vs_humans"] = compare(best["win_rate"], humans, "free")
        alt["vs_humans"] = compare(alt["win_rate"], humans, "human_style")

    out = {"draft": did, "set": set_code, "pool_size": len(pool), "proposed": best,
           "human_style": alt if alt["lane"] != best["lane"] else None}
    safe = "".join(ch if ch.isalnum() else "_" for ch in did).strip("_")
    path = CAPTURES / f"deck_{safe}.json"
    path.write_text(json.dumps(out, indent=2))

    def show(label, deck):
        print(f"\n{label}: {deck['lane']}  ({sum(e['count'] for e in deck['spells'])} spells, "
              f"{sum(e['count'] for e in deck['lands'])} lands, {deck['creatures']} creatures)")
        print("  curve: " + "  ".join(f"{k}:{deck['curve'][k]}" for k in sorted(deck["curve"], key=lambda k: (k == "6+", k))))
        for e in deck["spells"]:
            flag = "" if e["castable"] >= 0.8 else f"   (castable {e['castable']:.0%})"
            print(f"   {e['count']}x {e['name']:<38} {e['cost']:<12}{flag}")
        print("  lands: " + ", ".join(f"{e['count']} {e['name']}" for e in deck["lands"]))
        v = deck.get("vs_humans")
        print(f"  rating: {100 * deck['win_rate']:.1f}% judged win rate (spells castable on average "
              f"{deck['mean_castability']:.0%})")
        if v:
            print(f"  vs human decks built the same way: average {100 * v['avg']:.1f}%, winning record "
                  f"{100 * v['avg_winning']:.1f}%, 7 wins {100 * v['avg_7_wins']:.1f}%")
            print(f"  better than {v['percentile']:.0f}% of human decks and {v['percentile_vs_winning']:.0f}% of "
                  f"winning-record decks (n={v['n']})")
    show("Proposed deck", best)
    if out["human_style"]:
        show("Alternative (the colors humans would most likely pick)", alt)
    if not humans:
        print(f"\n(No comparison data: run scripts/compare_decks.py --set {set_code} first.)")
    print(f"\nSaved {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
