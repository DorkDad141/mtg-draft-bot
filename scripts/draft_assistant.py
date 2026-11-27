"""Live draft assistant for MTG Arena: ranks every pack and blacks out all but the top N.

Live (start this, then start your draft in Arena; Arena in windowed or borderless mode):
    python scripts/draft_assistant.py --top 3

Rehearse without a paid draft, using a real 17Lands draft of any set as a fake live log,
with a "fake Arena" window showing the pack so the overlay can be tested too:
    python scripts/draft_assistant.py --top 3 --sim-17lands TRK --fake-screen
    python scripts/draft_assistant.py --top 3 --sim-logs logs/archive/Player_2026-05-09_*.gz

Options: --no-overlay (advisor panel only), --console (no windows at all),
--opacity 0.9 (see-through blackout), --interval 1.0 (seconds between screen checks),
--no-shield (only black out located cards; by default the whole pack area stays black
from your pick until the next pack is ranked, so unranked cards are never shown).
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mtgdraft.assistant import DraftAssistant, grab_full_screen  # noqa: E402
from mtgdraft.carddb import ArenaCardDB  # noqa: E402
from mtgdraft.reader import ArenaLogReader, default_arena_log_path  # noqa: E402
from mtgdraft.reader.archive import archive_logs, open_log  # noqa: E402
from mtgdraft.reader.simulate import notify_line, pack_in_line, steps_from_17lands  # noqa: E402

SIM_LOG = REPO_ROOT / "data" / "sim" / "Player.log"


def sim_steps_17lands(db: ArenaCardDB, set_code: str, event: str, index: int, delay: float,
                      quick: bool = False):
    import pandas as pd
    from mtgdraft.model.games import split_label
    raw = REPO_ROOT / "data" / "raw" / f"draft_data_public.{set_code}.{event}.csv.gz"
    if not raw.exists():
        sys.exit(f"{raw.name} not found. Run: python scripts/download_17lands.py --set {set_code} "
                 f"--event {event} --types draft")
    wanted, rows = None, []
    seen_test = []
    for chunk in pd.read_csv(raw, chunksize=20000):
        lab = split_label(chunk["draft_id"].to_numpy())
        chunk = chunk[lab == 2]                       # held-out test drafts only
        for d in chunk["draft_id"].unique():
            if d not in seen_test:
                seen_test.append(d)
        if wanted is None and len(seen_test) > index:
            wanted = seen_test[index]
        if wanted is not None:
            part = chunk[chunk["draft_id"] == wanted]
            rows += part.to_dict("records")
            if len(seen_test) > index + 1 and part.empty:
                break
    if not rows:
        sys.exit("No held-out draft found.")
    grp_of = {}
    for r in rows:
        for k in r:
            if k.startswith("pack_card_") and k[10:] not in grp_of:
                g = db.grp_for_name(k[10:], set_code)
                if g is not None:
                    grp_of[k[10:]] = g
    event_name = f"QuickDraft_{set_code}_20990101" if quick else f"{event}_{set_code}_SIM"
    draft_id, steps = steps_from_17lands(rows, grp_of, event_name, delay, quick=quick)
    print(f"Simulating 17Lands {set_code} {event} test draft #{index} ({wanted[:8]}, {len(rows)} picks, "
          f"{rows[0]['event_match_wins']}-{rows[0]['event_match_losses']} record)")
    return steps


def sim_steps_logs(paths, draft_prefix, delay):
    from mtgdraft.reader.simulate import pick_line, pool_lines, queue_line
    src = ArenaLogReader()
    for p in paths:
        src.feed(open_log(p))
    src.finalize()
    drafts = list(src.drafts.values())
    if draft_prefix:
        drafts = [d for d in drafts if d.draft_id.startswith(draft_prefix)]
    if not drafts:
        sys.exit("No matching draft in those logs.")
    d = max(reversed(drafts), key=lambda d: len(d.picks) + len(d.inferred))
    steps = [(0.0, queue_line(d.event_name or "Unknown_Draft"))]
    for slot in sorted(set(d.packs) | set(d.picks)):
        if slot in d.packs:
            steps.append((0.0, notify_line(d.draft_id, slot, d.packs[slot])))
        c = d.pick_at(slot)
        if c is not None:
            steps.append((delay, pick_line(d.draft_id, slot, c)))
    if d.final_pool:
        steps += [(0.0, line) for line in pool_lines(d.event_name or "Unknown_Draft", d.final_pool)]
    print(f"Simulating recorded draft {d.summary()}")
    return steps


def feed_sim(steps, ui, db, fake_screen: bool, stop: threading.Event):
    """Write simulated log lines to SIM_LOG at a live-like pace (and show packs on the fake screen)."""
    from mtgdraft.overlay.images import CardImageCache
    cache = CardImageCache() if fake_screen else None
    SIM_LOG.parent.mkdir(parents=True, exist_ok=True)
    SIM_LOG.write_text("", encoding="utf-8")
    time.sleep(0.5)
    with open(SIM_LOG, "a", encoding="utf-8", newline="\n") as f:
        for delay, line in steps:
            if stop.is_set():
                return
            time.sleep(delay)
            cards = pack_in_line(line) if fake_screen else None
            if cards:
                paths = []
                for g in cards:
                    c = db.get(g)
                    p = cache.get(g, db.name(g), c.set_code if c else None, "normal")
                    paths.append(str(p) if p else "")
                time.sleep(0.5)          # Arena: server round trip + the new pack sliding in
                ui.show_fake_pack(paths)
                if os.environ.get("MTGDRAFT_TRACE"):
                    print(f"[trace {time.time():.3f}] fake screen: new pack shown", flush=True)
                time.sleep(0.4)          # worst case: the pack is on screen before its log line
            f.write(line + "\n")
            f.flush()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--top", type=int, default=3, help="How many cards to leave visible (default 3)")
    ap.add_argument("--log", type=Path, help="Arena Player.log to follow (default: Arena's own)")
    ap.add_argument("--no-overlay", action="store_true", help="Advisor panel only, no screen blackout")
    ap.add_argument("--console", action="store_true", help="Text only, no windows")
    ap.add_argument("--opacity", type=float, default=1.0, help="Blackout opacity (1.0 solid, 0.85 dim)")
    ap.add_argument("--interval", type=float, default=1.0, help="Seconds between screen checks")
    ap.add_argument("--no-record", action="store_true",
                    help="Live drafts: don't save screenshots and the log to captures/ (recording is on by default)")
    ap.add_argument("--record", action="store_true", help="Simulations: record to captures/ too")
    ap.add_argument("--no-shield", action="store_true",
                    help="Black out only the located non-top cards, instead of covering the whole pack "
                         "area from the moment you pick until the next pack is ranked")
    ap.add_argument("--no-auto-train", action="store_true", help="Don't download/train a model for a new set")
    ap.add_argument("--sim-17lands", metavar="SET", help="Simulate a held-out 17Lands draft of SET")
    ap.add_argument("--sim-event", default="PremierDraft")
    ap.add_argument("--sim-index", type=int, default=0, help="Which held-out draft to simulate")
    ap.add_argument("--sim-logs", nargs="+", type=Path, help="Simulate a draft from recorded Arena logs")
    ap.add_argument("--sim-draft", help="Draft ID prefix for --sim-logs")
    ap.add_argument("--sim-delay", type=float, default=2.0, help="Seconds per pick in simulations")
    ap.add_argument("--sim-quick", action="store_true",
                    help="--sim-17lands: write Quick Draft (bot draft) log lines instead of Premier Draft ones")
    ap.add_argument("--fake-screen", action="store_true", help="Simulations: show packs in a fake Arena window")
    ap.add_argument("--capture-visible", action="store_true", help="Let screenshots see the overlay (testing)")
    args = ap.parse_args()

    simulating = bool(args.sim_17lands or args.sim_logs)
    if args.fake_screen and not simulating:
        ap.error("--fake-screen only works with --sim-17lands or --sim-logs")
    db = ArenaCardDB()

    ui = None
    if not args.console:
        from mtgdraft.overlay.window import OverlayUI, make_dpi_aware
        make_dpi_aware()
        # Live: the card finder captures Arena's own window, so the overlay must NOT be excluded
        # from capture (excluded windows are hidden over Arena). Rehearsals capture the whole
        # screen (the fake Arena window), so there the overlay is excluded from capture.
        ui = OverlayUI(args.top, blackout=not args.no_overlay, opacity=args.opacity,
                       exclude_from_capture=simulating and not args.capture_visible,
                       # live: show the blackout only while Arena (or the advisor) is the active window
                       only_over=None if simulating else "MTGA")

    if simulating:
        log_path = SIM_LOG
        steps = (sim_steps_17lands(db, args.sim_17lands, args.sim_event, args.sim_index, args.sim_delay,
                                   args.sim_quick)
                 if args.sim_17lands else sim_steps_logs(args.sim_logs, args.sim_draft, args.sim_delay))
    else:
        log_path = args.log or default_arena_log_path()
        try:
            saved = [r for r in archive_logs() if r.action in ("new", "updated")]
            if saved:
                print(f"Archived {len(saved)} earlier Arena log(s) to logs/archive/")
        except OSError as e:
            print(f"(log archive skipped: {e})")
    recorder = None
    if (args.record or not simulating) and not args.no_record:
        from mtgdraft.recorder import SessionRecorder
        recorder = SessionRecorder(log_path=log_path, settings={
            "top": args.top, "shield": not args.no_shield, "opacity": args.opacity, "interval": args.interval,
            "overlay": ui is not None and not args.no_overlay, "log": str(log_path),
            "simulation": args.sim_17lands or (args.sim_logs and [str(x) for x in args.sim_logs]) or None})
    assistant = DraftAssistant(db, top_n=args.top, ui=ui, interval=args.interval,
                               auto_train=not args.no_auto_train, shield=not args.no_shield, recorder=recorder,
                               # rehearsals use a fake window: don't save its pack area for real drafts
                               **({"region_file": None, "grab": grab_full_screen} if simulating else {}))
    print(f"Following {log_path}. Top {args.top} cards stay visible. Close the advisor window or Ctrl+C to stop.")

    stop = threading.Event()
    reader = ArenaLogReader(on_update=assistant.on_update)
    if simulating:
        # Start from an empty simulated log so the follower only sees this run.
        SIM_LOG.parent.mkdir(parents=True, exist_ok=True)
        SIM_LOG.write_text("", encoding="utf-8")
    follower = threading.Thread(target=reader.follow, args=(log_path, 0.05, stop.is_set), daemon=True)
    follower.start()
    if simulating:
        threading.Thread(target=feed_sim, args=(steps, ui, db, args.fake_screen, stop), daemon=True).start()

    try:
        if ui is not None:
            ui.run()
        else:
            while follower.is_alive():
                time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        assistant.stop.set()
        if recorder is not None:
            recorder.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
