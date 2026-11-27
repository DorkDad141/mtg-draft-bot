"""Propose a 40-card deck from a recorded draft's pool, rate it, and compare it with human decks.

  * deck: built by the same scorer that ranked the picks (ManaAwareScorer.build_deck): the
    colors it rates most likely, the best 23 spells by its castability-weighted values,
    the nonbasic lands its land rule plays (lands making 2+ of the deck's colors, plus
    utility lands when the model learned from real decklists), and basics from the same
    mana base. No separate deck-building logic.
  * rating: the independent judge (a win-rate model trained only on held-out drafts the
    scorer never saw), with each spell's uncastable share valued as a dead card, so the
    scorer is not grading its own deck. Neutral controls (no skill or rank): it rates the
    deck, not the player.
  * comparison: human pools from 17Lands drafts the judge never saw, built by the same
    scorer and rated by the same judge (scripts/compare_decks.py, which this reads): the
    average deck, the average deck with a winning match record, 7-win decks, and your
    deck's percentile.

    python scripts/propose_deck.py                 # newest recorded draft
    python scripts/propose_deck.py --draft QuickDraft_TRK
Writes captures/deck_<draft>.json, which draft_viewer.py shows as a final "Your deck" slide.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mtgdraft.carddb import ArenaCardDB  # noqa: E402
from mtgdraft.model.scorer import load_scorer  # noqa: E402
from scripts.ab_test import MODELS, DeckBuilder, rate_deck, train_judge  # noqa: E402
from scripts.draft_viewer import CAPTURES, load_sessions  # noqa: E402


def describe(deck: dict, info: dict) -> dict:
    """Card lists for display: spells by mana value (with castability), then lands."""
    p_of = {}
    for n, p in zip(deck["spells"], deck["castable"]):
        p_of[n] = min(p, p_of.get(n, 1.0))
    spells = []
    for n, c in Counter(deck["spells"]).items():
        ci = info.get(n)
        spells.append({"name": n, "count": c, "mv": ci.mana_value if ci else None, "colors": ci.colors if ci else "",
                       "cost": ci.mana_cost if ci else "", "type": " ".join(ci.types) if ci else "",
                       "castable": round(p_of[n], 2)})
    spells.sort(key=lambda e: (e["mv"] if e["mv"] is not None else 99, e["name"]))
    lands = [{"name": n, "count": c} for n, c in sorted(deck["basics"].items())]
    lands += [{"name": n, "count": c} for n, c in sorted(Counter(deck["lands"]).items())]
    curve = {}
    for e in spells:
        k = "6+" if (e["mv"] or 0) >= 6 else str(e["mv"] or 0)
        curve[k] = curve.get(k, 0) + e["count"]
    return {"lane": deck["lane_name"], "spells": spells, "lands": lands,
            "creatures": sum(e["count"] for e in spells if "Creature" in e["type"]), "curve": curve,
            "mean_castability": round(float(np.mean(deck["castable"])) if deck["castable"] else 1.0, 3)}


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
    pool = list(d["complete"]["names"]) if d["complete"] else [p["name"] for _, p in sorted(d["picks"].items())]
    set_code = next(iter(d["packs"].values()))["set"]
    info = ArenaCardDB().by_name()
    scorer = load_scorer(set_code, args.event, card_info=info)
    if scorer is None or not hasattr(scorer, "build_deck"):
        sys.exit(f"No mana-aware model for {set_code}.")
    built = scorer.build_deck(pool)
    deck = describe(built, info)
    judge = train_judge(set_code, args.event, scorer.card_names)
    rater = DeckBuilder(judge, scorer.card_names, info)
    deck["win_rate"] = float(1 / (1 + np.exp(-rate_deck(rater, built["spells"], built["castable"],
                                                         built["lands"], built["basics"]))))
    cmp_path = MODELS / f"compare_decks.{set_code}.{args.event}.json"
    humans = json.loads(cmp_path.read_text()).get("human_decks") if cmp_path.exists() else None
    if humans and "scorer_built" in humans[0]:
        deck["vs_humans"] = compare(deck["win_rate"], humans, "scorer_built")

    out = {"draft": did, "set": set_code, "pool_size": len(pool), "proposed": deck, "human_style": None}
    safe = "".join(ch if ch.isalnum() else "_" for ch in did).strip("_")
    path = CAPTURES / f"deck_{safe}.json"
    path.write_text(json.dumps(out, indent=2))

    print(f"\nProposed deck: {deck['lane']}  ({sum(e['count'] for e in deck['spells'])} spells, "
          f"{sum(e['count'] for e in deck['lands'])} lands, {deck['creatures']} creatures)")
    print("  curve: " + "  ".join(f"{k}:{deck['curve'][k]}" for k in sorted(deck["curve"], key=lambda k: (k == "6+", k))))
    for e in deck["spells"]:
        flag = "" if e["castable"] >= 0.8 else f"   (castable {e['castable']:.0%})"
        print(f"   {e['count']}x {e['name']:<38} {e['cost']:<12}{flag}")
    print("  lands: " + ", ".join(f"{e['count']} {e['name']}" for e in deck["lands"]))
    print(f"  rating: {100 * deck['win_rate']:.1f}% judged win rate (spells castable on average "
          f"{deck['mean_castability']:.0%})")
    v = deck.get("vs_humans")
    if v:
        print(f"  vs human decks built and rated the same way: average {100 * v['avg']:.1f}%, winning record "
              f"{100 * v['avg_winning']:.1f}%, 7 wins {100 * v['avg_7_wins']:.1f}%")
        print(f"  better than {v['percentile']:.0f}% of human decks and {v['percentile_vs_winning']:.0f}% of "
              f"winning-record decks (n={v['n']})")
    else:
        print(f"\n(No comparison data: run scripts/compare_decks.py --set {set_code} first.)")
    print(f"\nSaved {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
