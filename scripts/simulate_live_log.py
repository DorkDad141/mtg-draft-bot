"""Simulate a live Arena draft from recorded logs, and watch the bot rank each pack.

Takes a draft from recorded Player.log files and writes it, one event at a time,
into a fresh simulated Player.log in the real Arena line format (Draft.Notify for
each pack, EventPlayerDraftMakePick for each pick, then the final card pool). The
live log reader follows that file exactly as it would follow Arena's, and every
pack is printed with each card's rank and score from the current model, followed
by the pick that was actually made and where the bot ranked it.

Picks come from the log when it has them, otherwise from the final pool (inferred).
With --bot-picks, picks that are still unknown are made by the bot itself, so the
pool (and the synergy scores) keep building. The packs you see later in the draft
still reflect the real picks, so treat that mode as approximate.

Examples:
    python scripts/simulate_live_log.py logs/Player-prev.log logs/Player.log --list
    python scripts/simulate_live_log.py logs/Player-prev.log logs/Player.log --draft f283
    python scripts/simulate_live_log.py logs/Player-prev.log logs/Player.log --draft 2165 --bot-picks --delay 0
    # Drive the real live viewer from a second terminal instead of watching here:
    python scripts/simulate_live_log.py logs/Player.log --no-watch --delay 2
    python scripts/show_draft.py --live data/sim/Player.log
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mtgdraft.carddb import ArenaCardDB  # noqa: E402
from mtgdraft.model.scorer import PickScorer, load_scorer  # noqa: E402
from mtgdraft.reader import ArenaLogReader, DraftState  # noqa: E402
from mtgdraft.reader.arena_log import set_and_event  # noqa: E402

DEFAULT_OUT = REPO_ROOT / "data" / "sim" / "Player.log"
PREFIX = "[UnityCrossThreadLogger]"


# ---- building the simulated log ------------------------------------------------
def queue_line(event_name: str) -> str:
    j = {"fromSceneName": "EventLanding", "toSceneName": "TableDraftQueue", "initiator": "System",
         "context": f"Entering table draft queue: {event_name}"}
    return f"{PREFIX}Client.SceneChange {json.dumps(j)}"


def notify_line(draft_id: str, slot, cards) -> str:
    j = {"draftId": draft_id, "SelfPick": slot[1], "SelfPack": slot[0],
         "PackCards": ",".join(str(c) for c in cards)}
    return f"{PREFIX}Draft.Notify {json.dumps(j, separators=(',', ':'))}"


def pick_line(draft_id: str, slot, card: int) -> str:
    req = {"DraftId": draft_id, "GrpIds": [card], "Pack": slot[0], "Pick": slot[1]}
    j = {"id": str(uuid.uuid4()), "request": json.dumps(req, separators=(',', ':'))}
    return f"{PREFIX}==> EventPlayerDraftMakePick {json.dumps(j, separators=(',', ':'))}"


def pool_lines(event_name: str, pool: list[int]) -> list[str]:
    rid = str(uuid.uuid4())
    course = {"CourseId": str(uuid.uuid4()), "InternalEventName": event_name,
              "CurrentModule": "DeckSelect", "ModulePayload": "", "CardPool": pool}
    return [f"<== DraftCompleteDraft({rid})", json.dumps(course, separators=(',', ':'))]


# ---- watching ------------------------------------------------------------------
class Watcher:
    """Prints each pack ranked by the bot as the live reader sees it."""

    def __init__(self, db: ArenaCardDB, top: int | None, pick_source: dict):
        self.db = db
        self.top = top
        self.pick_source = pick_source      # slot -> "logged" / "inferred" / "bot"
        self.ranked: dict = {}              # slot -> [(name, score), ...]
        self.agree = self.compared = 0
        self.lock = threading.Lock()

    def scorer(self, d: DraftState) -> PickScorer | None:
        set_code, event = set_and_event(d, lambda g: (self.db.get(g).set_code if self.db.get(g) else "?"))
        return load_scorer(set_code, event)

    def __call__(self, d: DraftState, kind: str) -> None:
        with self.lock:
            if kind == "pack":
                self.on_pack(d)
            elif kind == "pick":
                self.on_pick(d)
            elif kind == "pool":
                print(f"\nDraft complete: final pool of {len(d.final_pool or [])} cards received.", flush=True)

    def on_pack(self, d: DraftState) -> None:
        slot = d.current_slot()
        if slot is None:
            return
        pack = self.db.names(d.packs[slot])
        pool = self.db.names(d.pool_before(slot))
        sc = self.scorer(d)
        print(f"\nP{slot[0]}P{slot[1]}  ({len(pack)} cards, pool {len(pool)})", flush=True)
        if sc is None:
            print("  (no model for this set) " + ", ".join(pack), flush=True)
            return
        ranked = sc.rank(pack, pool)
        self.ranked[slot] = ranked
        shown = ranked if self.top is None else ranked[:self.top]
        for i, (name, s) in enumerate(shown, 1):
            score = "   --  " if s != s else f"{s:+.3f}"
            print(f"  {i:>2}. {score}  {name}", flush=True)
        if len(shown) < len(ranked):
            print(f"      ... {len(ranked) - len(shown)} more", flush=True)

    def on_pick(self, d: DraftState) -> None:
        slot = max(d.picks)                      # the pick just made
        name = self.db.name(d.picks[slot])
        src = self.pick_source.get(slot, "logged")
        label = {"logged": "your pick", "inferred": "your pick (inferred from final pool)",
                 "bot": "bot's pick (your pick unknown)"}[src]
        ranked = self.ranked.get(slot)
        if ranked is None:
            print(f"\nP{slot[0]}P{slot[1]}  pack not in the log; {label}: {name}", flush=True)
            return
        names = [n for n, _ in ranked]
        rank = names.index(name) + 1 if name in names else None
        where = f"bot rank #{rank} of {len(names)}" if rank else "not in the ranked pack"
        mark = "  <- matches bot" if rank == 1 and src != "bot" and len(names) > 1 else ""
        print(f"  => {label}: {name}  ({where}){mark}", flush=True)
        if src != "bot" and len(names) > 1:
            self.compared += 1
            self.agree += rank == 1


# ---- main ----------------------------------------------------------------------
def choose_draft(drafts: dict[str, DraftState], prefix: str | None) -> DraftState:
    if prefix:
        hits = [d for k, d in drafts.items() if k.startswith(prefix)]
        if len(hits) != 1:
            sys.exit(f"--draft {prefix!r} matches {len(hits)} drafts; use --list")
        return hits[0]
    # Default: the draft with the most known picks (ties: the last one in the logs).
    return max(reversed(list(drafts.values())), key=lambda d: len(d.picks) + len(d.inferred))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+", help="Recorded Player-prev.log / Player.log, oldest first")
    ap.add_argument("--list", action="store_true", help="List drafts in the logs and exit")
    ap.add_argument("--draft", help="Draft ID prefix (default: the draft with the most known picks)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"Simulated log (default {DEFAULT_OUT.relative_to(REPO_ROOT)})")
    ap.add_argument("--delay", type=float, default=1.0, help="Seconds between a pack and its pick (default 1)")
    ap.add_argument("--top", type=int, default=None, help="Show only the top N cards per pack (default: all)")
    ap.add_argument("--bot-picks", action="store_true", help="Let the bot make picks that are unknown")
    ap.add_argument("--no-watch", action="store_true", help="Only write the simulated log")
    args = ap.parse_args()

    db = ArenaCardDB()
    src = ArenaLogReader()
    for p in args.logs:
        src.feed(open(p, encoding="utf-8", errors="replace"))
    src.finalize()
    if not src.drafts:
        sys.exit("No drafts found in those logs.")

    if args.list:
        for d in src.drafts.values():
            print(d.summary())
        return 0

    d = choose_draft(src.drafts, args.draft)
    event_name = d.event_name or "Unknown_Draft"
    set_code, event = set_and_event(d, lambda g: (db.get(g).set_code if db.get(g) else "?"))
    bot = load_scorer(set_code, event)
    print(f"Simulating {d.summary()}")
    print(f"Set {set_code}, {event}; model: {'yes' if bot else 'NONE (train one with train_deck_model.py)'}")
    print(f"Writing {args.out}")

    # Plan the event sequence and pick sources.
    pick_source: dict = {}
    steps: list[tuple[float, str]] = [(0.0, queue_line(event_name))]
    sim_pool: list[int] = []
    for slot in sorted(set(d.packs) | set(d.picks)):
        if slot in d.packs:
            steps.append((0.0, notify_line(d.draft_id, slot, d.packs[slot])))
        card = d.pick_at(slot)
        if card is not None:
            pick_source[slot] = "logged" if slot in d.picks else "inferred"
        elif args.bot_picks and bot and slot in d.packs:
            ranked = bot.rank(db.names(d.packs[slot]), db.names(sim_pool))
            by_name = {db.name(g): g for g in d.packs[slot]}
            card = by_name[ranked[0][0]]
            pick_source[slot] = "bot"
        if card is not None:
            sim_pool.append(card)
            steps.append((args.delay, pick_line(d.draft_id, slot, card)))
    final_pool = d.final_pool if d.final_pool and not any(v == "bot" for v in pick_source.values()) else sim_pool
    if final_pool:
        for line in pool_lines(event_name, final_pool):
            steps.append((0.0, line))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("", encoding="utf-8")

    watcher = stop = thread = None
    poll = 0.05
    if not args.no_watch:
        watcher = Watcher(db, args.top, pick_source)
        stop = threading.Event()
        live = ArenaLogReader(on_update=watcher)
        thread = threading.Thread(target=live.follow, args=(args.out, poll, stop.is_set), daemon=True)
        thread.start()

    with open(args.out, "a", encoding="utf-8", newline="\n") as f:
        for delay, line in steps:
            time.sleep(max(delay, 3 * poll if watcher else 0.0))
            f.write(line + "\n")
            f.flush()

    if watcher:
        time.sleep(10 * poll)
        stop.set()
        thread.join(timeout=2)
        n = sum(1 for v in pick_source.values() if v == "bot")
        print()
        if watcher.compared:
            print(f"Bot's #1 matched the real pick on {watcher.agree}/{watcher.compared} picks "
                  f"({watcher.agree / watcher.compared:.0%}), excluding single-card packs.")
        if n:
            print(f"The bot made {n} picks that were unknown in the log.")
        unknown = [s for s in d.packs if s not in pick_source]
        if unknown:
            print(f"{len(unknown)} picks unknown and skipped (use --bot-picks to let the bot pick).")
    print(f"Simulated log written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
