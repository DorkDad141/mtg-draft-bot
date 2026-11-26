"""Parse Arena Player.log files and replay each draft through the pick scorer.

Prints every pick with the pack size, your pick, and the scorer's top cards, then
runs consistency checks (picks are in their packs, pack sizes count down, logged
plus inferred picks rebuild the final pool).

Pass logs oldest first so drafts that span an Arena restart are joined:
    python scripts/show_draft.py logs/Player-prev.log logs/Player.log
    python scripts/show_draft.py --live          # follow the running Arena log
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mtgdraft.carddb import ArenaCardDB  # noqa: E402
from mtgdraft.model.fm import FactorizationMachine  # noqa: E402
from mtgdraft.model.scorer import PickScorer  # noqa: E402
from mtgdraft.reader import ArenaLogReader, DraftState, default_arena_log_path  # noqa: E402
from mtgdraft.reader.arena_log import set_and_event  # noqa: E402

_scorers: dict[tuple, PickScorer | None] = {}


def scorer_for(set_code: str | None, event: str) -> PickScorer | None:
    key = (set_code, event)
    if key not in _scorers:
        path = REPO_ROOT / "data" / "models" / f"fm.{set_code}.{event}.npz"
        if set_code and path.exists():
            fm, names = FactorizationMachine.load(path)
            _scorers[key] = PickScorer(fm, names["cards"])
        else:
            _scorers[key] = None
    return _scorers[key]


def _fmt(name: str, score: float) -> str:
    return f"{name} (not scored)" if score != score else f"{name} {score:+.2f}"


def check(d: DraftState) -> list[str]:
    problems = []
    for slot, card in d.picks.items():
        if slot in d.packs and card not in d.packs[slot]:
            problems.append(f"pick {slot} not in its pack")
    for (pack, pick), cards in d.packs.items():
        prev = d.packs.get((pack, pick - 1))
        if prev is not None and len(cards) != len(prev) - 1:
            problems.append(f"pack size jumps at {(pack, pick)}: {len(prev)} -> {len(cards)}")
    if d.final_pool:
        rebuilt = Counter(c for s in d.packs if (c := d.pick_at(s)) is not None)
        extra = rebuilt - Counter(d.final_pool)
        if extra:
            problems.append(f"{sum(extra.values())} picks not in final pool")
    return problems


def show(d: DraftState, db: ArenaCardDB, top: int) -> None:
    set_code, event = set_and_event(d, lambda g: (db.get(g).set_code if db.get(g) else "?"))
    sc = scorer_for(set_code, event)
    print(f"\n=== {d.summary()}  [set {set_code}, {event}, model {'yes' if sc else 'none'}]")
    agree = n = 0
    for slot in sorted(d.packs):
        pack = db.names(d.packs[slot])
        mine = d.pick_at(slot)
        tag = "" if slot in d.picks else (" (inferred)" if slot in d.inferred else "")
        mine_name = db.name(mine) + tag if mine is not None else "?"
        line = f"  P{slot[0]}P{slot[1]:<2} {len(pack):>2} cards  pick: {mine_name:<40}"
        if sc:
            ranked = sc.rank(pack, db.names(d.pool_before(slot)))
            if mine is not None and len(pack) > 1:
                n += 1
                agree += ranked[0][0] == db.name(mine)
            line += " | bot: " + ", ".join(_fmt(nm, s) for nm, s in ranked[:top])
        print(line)
    if n:
        print(f"  bot's top pick matched yours on {agree}/{n} picks ({agree / n:.0%})")
    problems = check(d)
    print("  checks: " + ("OK" if not problems else "; ".join(problems)))
    missing = d.missing_picks()
    if missing:
        print(f"  picks unknown for {len(missing)} slots (not logged and not inferable)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="*", help="Player-prev.log / Player.log, oldest first")
    ap.add_argument("--live", action="store_true", help="Follow the running Arena Player.log")
    ap.add_argument("--top", type=int, default=3)
    args = ap.parse_args()
    db = ArenaCardDB()

    if args.live:
        path = Path(args.logs[0]) if args.logs else default_arena_log_path()
        print(f"Following {path} (Ctrl+C to stop)")

        def on_update(d: DraftState, kind: str) -> None:
            if kind != "pack":
                return
            slot = d.current_slot()
            if slot is None:
                return
            set_code, event = set_and_event(d, lambda g: (db.get(g).set_code if db.get(g) else "?"))
            sc = scorer_for(set_code, event)
            pack = db.names(d.packs[slot])
            print(f"\nP{slot[0]}P{slot[1]} ({len(pack)} cards)")
            if sc:
                for nm, s in sc.rank(pack, db.names(d.pool_before(slot)))[:args.top]:
                    print(f"  {_fmt(nm, s)}")
            else:
                print(f"  no model for {set_code} {event}: " + ", ".join(pack))

        try:
            ArenaLogReader(on_update=on_update).follow(path)
        except KeyboardInterrupt:
            pass
        return 0

    if not args.logs:
        ap.error("give log files, or --live")
    reader = ArenaLogReader()
    for p in args.logs:
        reader.feed(open(p, encoding="utf-8", errors="replace"))
    reader.finalize()
    if not reader.drafts:
        print("No drafts found.")
    for d in reader.drafts.values():
        show(d, db, args.top)
    return 0


if __name__ == "__main__":
    sys.exit(main())
