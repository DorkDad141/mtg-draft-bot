"""Live draft assistant: log reader -> deck-aware ranking -> advisor + blackout overlay.

Threads:
  * log follower (ArenaLogReader.follow) calls on_update for each pack / pick / pool
  * locator loop: while a pack is open, screenshots the screen about once a second,
    finds the pack's cards, and posts black boxes for all but the top N
  * shield (default): the moment a pick is logged, the whole pack area goes black, so
    the next pack is never seen unranked; once its cards are found, holes are cut for
    the top N only. The pack area is learned from where cards were found and saved
    (data/overlay_region.json), so even P1P1 of later drafts starts covered.
  * background jobs: card art prefetch for the set, and model download/training
    when the set has no model yet
  * the Tk main loop (OverlayUI) owns the windows
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .carddb import ArenaCardDB
from .model import scorer as scorer_mod
from .model.scorer import load_scorer
from .reader import DraftState
from .reader.arena_log import set_and_event

REPO_ROOT = Path(__file__).resolve().parents[1]
REGION_FILE = REPO_ROOT / "data" / "overlay_region.json"
FULL_COVER_SECONDS = 3.0     # cover the whole screen at most this long while the pack area is unknown
NOT_FOUND_SECONDS = 3.0      # lift the cover if no card is found this long (Arena hidden or minimized)


@dataclass
class Current:
    draft_id: str
    slot: tuple[int, int]
    pack: list[int]              # grpIds
    top_keys: set                # grpIds of the top N cards
    rank_of: dict                # grpId -> rank (1-based)
    set_code: str | None
    boxes: dict = None           # grpId -> last card boxes found for this pick (sticky)
    since: float = 0.0           # when the pack arrived


class DraftAssistant:
    def __init__(self, db: ArenaCardDB, top_n: int = 3, ui=None, interval: float = 1.0,
                 auto_train: bool = True, printer=print, grab=None, locate: bool = True,
                 shield: bool = True, screen_size: tuple[int, int] | None = None, recorder=None,
                 region_file: Path | None = REGION_FILE):
        self.db = db
        self.info = db.by_name()
        self.top_n = top_n
        self.ui = ui
        self.interval = interval
        self.auto_train = auto_train
        self.print = printer
        self.grab = grab or _grab_screen
        self.locate_enabled = locate and ui is not None and getattr(ui, "overlay", None) is not None
        self.shield = shield and self.locate_enabled
        self.recorder = recorder
        self.screen_size = screen_size
        self.region_file = region_file      # None: learn the pack area for this run only
        self.region = self._load_region()
        self._shield_since = None
        self.wake = threading.Event()
        self.current: Current | None = None
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self._training: set = set()
        self._prefetched: set = set()
        self._last_state: DraftState | None = None
        self.images = self.locator = None
        if self.locate_enabled:
            from .overlay.images import CardImageCache
            from .overlay.locate import CardLocator
            self.images, self.locator = CardImageCache(), CardLocator()
            threading.Thread(target=self._locate_loop, daemon=True).start()

    # ---- log events -------------------------------------------------------------
    def on_update(self, d: DraftState, kind: str) -> None:
        try:
            self._last_state = d
            if kind == "pack":
                self._on_pack(d)
            elif kind == "pick":
                self._on_pick(d)
            elif kind == "pool":
                self._on_complete(d)
        except Exception as e:  # keep following the log no matter what
            self.print(f"[assistant] error handling {kind}: {e!r}")

    def _set_of(self, g: int) -> str:
        c = self.db.get(g)
        return c.set_code if c else "?"

    def _scorer(self, set_code, event):
        sc = load_scorer(set_code, event, card_info=self.info)
        if sc is None and event != "PremierDraft":
            # Quick/Traditional drafts: the decks are the same game, so the set's Premier Draft
            # model ranks them well; only what the other drafters take differs.
            sc = load_scorer(set_code, "PremierDraft", card_info=self.info)
            if sc is not None:
                return sc
        if sc is None and set_code and self.auto_train and (set_code, event) not in self._training:
            self._training.add((set_code, event))
            threading.Thread(target=self._train, args=(set_code, event), daemon=True).start()
        return sc

    def _on_pack(self, d: DraftState) -> None:
        slot = d.current_slot()
        if slot is None:
            return
        self._raise_shield("Ranking this pack...")
        set_code, event = set_and_event(d, self._set_of)
        self._prefetch(set_code)
        pack = d.packs[slot]
        names = self.db.names(pack)
        pool = self.db.names(d.pool_before(slot))
        sc = self._scorer(set_code, event)
        header = f"P{slot[0]}P{slot[1]} - {set_code or '?'} {event} - {len(pack)} cards"
        if sc is None:
            msg = (f"No model for {set_code} {event} yet" +
                   (" - downloading 17Lands data and training (a minute or two)..." if self.auto_train else "."))
            self.print(f"\n{header}\n  {msg}")
            if self.ui:
                self.ui.show_pack(header, "", [(n, "", False) for n in names], msg)
            with self.lock:
                self.current = None
            return
        ranking = sc.explain(names, pool)
        lanes = sc.lanes(pool) if hasattr(sc, "lanes") else []
        top_names, rank_of_name = [], {}
        for i, r in enumerate(ranking, 1):
            rank_of_name.setdefault(r["name"], i)
            if r["score"] == r["score"] and len(top_names) < self.top_n and r["name"] not in top_names:
                top_names.append(r["name"])
        top_keys = {g for g, n in zip(pack, names) if n in top_names}
        rank_of = {g: rank_of_name.get(n) for g, n in zip(pack, names)}
        lanes_txt = ("Projected deck: " + "  ".join(f"{p} {w:.0%}" for p, w, _ in lanes)) if lanes else ""
        rows = []
        for i, r in enumerate(ranking, 1):
            score = "--" if r["score"] != r["score"] else f"{100 * 0.2475 * r['score']:+.2f}%"
            rows.append((f"{i:>2}. {r['name'][:30]}", score, r["name"] in top_names))
        self.print(f"\n{header}   {lanes_txt}")
        for left, right, top in rows[:max(self.top_n, 3)]:
            self.print(f"  {'*' if top else ' '} {left:<34}{right:>9}")
        if self.recorder:
            self.recorder.event("pack", draft_id=d.draft_id, slot=list(slot), set=set_code, event=event,
                                event_name=d.event_name, pack=list(pack), names=names, pool=pool,
                                ranking=[{"name": r["name"], "score": None if r["score"] != r["score"]
                                          else float(r["score"]), **({"lane": r["lane"]} if "lane" in r else {})}
                                         for r in ranking],
                                top=top_names, lanes=[[p, float(w)] for p, w, *_ in lanes])
        with self.lock:
            self.current = Current(d.draft_id, slot, list(pack), top_keys, rank_of, set_code, since=time.time())
        self.wake.set()
        if self.ui:
            self.ui.show_pack(header, lanes_txt, rows,
                              "Locating cards on screen..." if self.locate_enabled else "")

    def _on_pick(self, d: DraftState) -> None:
        slot = max(d.picks)
        with self.lock:
            cur = self.current
            if cur and cur.draft_id == d.draft_id and cur.slot == slot:
                self.current = None
        name = self.db.name(d.picks[slot])
        rank = cur.rank_of.get(d.picks[slot]) if cur and cur.slot == slot else None
        msg = f"P{slot[0]}P{slot[1]} picked {name}" + (f" (bot rank #{rank})" if rank else "")
        self.print("  -> " + msg)
        if self.recorder:
            self.recorder.event("pick", draft_id=d.draft_id, slot=list(slot), pick=d.picks[slot], name=name,
                                bot_rank=rank, inferred=slot in getattr(d, "inferred", ()))
        if self.ui:
            last_pick = slot[0] >= 3 and len(d.packs.get(slot, ())) <= 1
            if self.shield and not last_pick:
                self._raise_shield("Next pack...")
            else:
                self.ui.clear_boxes()
            self.ui.set_status(msg)

    def _on_complete(self, d: DraftState) -> None:
        with self.lock:
            self.current = None
        set_code, event = set_and_event(d, self._set_of)
        sc = self._scorer(set_code, event)
        pool = self.db.names(d.final_pool or [])
        lanes = sc.lanes(pool, top=3) if sc is not None and hasattr(sc, "lanes") else []
        lanes_txt = "Final pool leans: " + "  ".join(f"{p} {w:.0%} ({n} playables)" for p, w, n in lanes)
        self.print(f"\nDraft complete ({len(pool)} cards). {lanes_txt}")
        if self.recorder:
            self.recorder.event("complete", draft_id=d.draft_id, pool=list(d.final_pool or []), names=pool,
                                lanes=[[p, float(w), n] for p, w, n in lanes])
            self.recorder.save_log()
        if self.ui:
            self.ui.clear_boxes()
            self.ui.show_pack(f"Draft complete - {len(pool)} cards", lanes_txt,
                              [(n, "", False) for n in sorted(pool)], "Good luck!")

    # ---- shield -----------------------------------------------------------------
    def _raise_shield(self, message: str) -> None:
        """Black out the whole pack area now, before the next pack is located."""
        if not self.shield:
            return
        self._shield_since = time.time()
        self.ui.set_boxes([], [], cover=self.region or "full", message=message)

    def _screen_key(self) -> str:
        if self.screen_size is None:
            try:
                import ctypes
                u = ctypes.windll.user32
                self.screen_size = (u.GetSystemMetrics(0), u.GetSystemMetrics(1))
            except (AttributeError, OSError):
                self.screen_size = (0, 0)
        return f"{self.screen_size[0]}x{self.screen_size[1]}"

    def _load_region(self):
        if not self.shield or self.region_file is None:
            return None
        try:
            r = json.loads(self.region_file.read_text()).get(self._screen_key())
            return tuple(r) if r else None
        except (OSError, ValueError):
            return None

    def _learn_region(self, boxes) -> None:
        """Grow the pack area to cover every card box found (with a margin), and save it."""
        x0 = min(b[0] for b in boxes)
        y0 = min(b[1] for b in boxes)
        x1 = max(b[2] for b in boxes)
        y1 = max(b[3] for b in boxes)
        pad = int(0.06 * max(b[3] - b[1] for b in boxes))
        new = (max(0, x0 - pad), max(0, y0 - pad), x1 + pad, y1 + pad)
        old = self.region
        if old:
            new = (min(old[0], new[0]), min(old[1], new[1]), max(old[2], new[2]), max(old[3], new[3]))
        if new != old:
            self.region = new
            if self.region_file is None:
                return
            try:
                f = self.region_file
                data = json.loads(f.read_text()) if f.exists() else {}
                data[self._screen_key()] = list(new)
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_text(json.dumps(data))
            except (OSError, ValueError):
                pass

    # ---- background jobs ----------------------------------------------------------
    def _prefetch(self, set_code: str | None) -> None:
        if not self.locate_enabled or not set_code or set_code in self._prefetched:
            return
        self._prefetched.add(set_code)
        cards, seen = [], set()
        for c in self.db._cards.values():
            if c.set_code == set_code and c.rarity not in ("token",) and c.name not in seen:
                seen.add(c.name)
                cards.append((self.db.grp_for_name(c.name, set_code), c.name, set_code))

        def job():
            n = self.images.prefetch(cards)
            self.print(f"[assistant] card art ready for {n}/{len(cards)} {set_code} cards")
        threading.Thread(target=job, daemon=True).start()

    def _train(self, set_code: str, event: str) -> None:
        py = sys.executable
        steps = [[py, str(REPO_ROOT / "scripts" / "download_17lands.py"), "--set", set_code, "--event", event,
                  "--types", "draft", "game"],
                 [py, str(REPO_ROOT / "scripts" / "train_deck_model.py"), "--set", set_code, "--event", event,
                  "--quiet"]]
        for cmd in steps:
            self.print(f"[assistant] running: {' '.join(Path(c).name if i < 2 else c for i, c in enumerate(cmd))}")
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                tail = (r.stdout + r.stderr).strip().splitlines()[-3:]
                self.print(f"[assistant] failed: {' | '.join(tail)}")
                if self.ui:
                    self.ui.set_status(f"Could not build a model for {set_code} {event}: {' '.join(tail)[:200]}")
                return
        scorer_mod._cache.clear()
        self.print(f"[assistant] model ready for {set_code} {event}")
        if self._last_state is not None:
            self._on_pack(self._last_state)

    def _locate_loop(self) -> None:
        while not self.stop.is_set():
            with self.lock:
                cur = self.current
            if cur is None:
                # Don't leave the whole screen black if no pack follows (e.g. a full cover at P1P1).
                if (self.region is None and self._shield_since
                        and time.time() - self._shield_since > FULL_COVER_SECONDS):
                    self._shield_since = None
                    self.ui.clear_boxes()
                if self.wake.wait(0.1):
                    self.wake.clear()
                continue
            t0 = time.time()
            try:
                counts = {}
                for g in cur.pack:
                    counts[g] = counts.get(g, 0) + 1
                cards = []
                for g, n in counts.items():
                    p = self.images.get(g, self.db.name(g), cur.set_code)
                    if p is not None:
                        cards.append((g, str(p), n))
                screen = self.grab()
                found = self.locator.locate(screen, cards)
            except Exception as e:
                self.print(f"[assistant] locate error: {e!r}")
                found = []
            with self.lock:
                still = self.current is cur
            if still:
                # Sticky boxes: keep a card's last box until it is found again. Cards rarely move
                # during a pick, and this avoids flicker if the overlay ever shows up in screenshots.
                if cur.boxes is None:
                    cur.boxes = {}
                fresh: dict = {}
                for f in found:
                    fresh.setdefault(f.key, []).append(f)
                cur.boxes.update(fresh)
                found = [f for fs in cur.boxes.values() for f in fs]
                sig = (cur.slot, len(found))
                if sig != getattr(self, "_last_sig", None):
                    self._last_sig = sig
                    self.print(f"  [overlay] P{cur.slot[0]}P{cur.slot[1]}: found {len(found)}/{len(cur.pack)} "
                               f"cards on screen ({(time.time() - t0) * 1000:.0f} ms)")
                badges = [(f.card_box, f"#{cur.rank_of.get(f.key)}") for f in found if f.key in cur.top_keys]
                waited = time.time() - cur.since
                drawn = None
                if not self.shield:
                    drawn = {"black": [f.card_box for f in found if f.key not in cur.top_keys], "badges": badges}
                    self.ui.set_boxes(drawn["black"], badges)
                elif found:
                    self._learn_region([f.card_box for f in found])
                    holes = [b for b, _ in badges] if cur.top_keys else [f.card_box for f in found]
                    drawn = {"cover": self.region, "holes": holes, "badges": badges}
                    self.ui.set_boxes([], badges, cover=self.region, holes=holes)
                if self.recorder and drawn is not None and found:
                    self.recorder.frame(cur.draft_id, cur.slot, screen, found, cur.top_keys, drawn,
                                        getattr(self.ui, "advisor_rect", None), getattr(self.ui, "opacity", 1.0))
                elif waited > NOT_FOUND_SECONDS or (self.region is None and waited > FULL_COVER_SECONDS):
                    self.ui.clear_boxes()           # pack not on screen: don't keep a black area up
                tops_found = len({f.key for f in found if f.key in cur.top_keys})
                self.ui.set_status(f"Found {len(found)}/{len(cur.pack)} cards on screen; "
                                   f"top {len(cur.top_keys)} visible: {tops_found}"
                                   + ("" if found else "  (is the pack on screen? Arena must not be minimized)"))
            # Re-check about once a second, but at once when a new pack arrives.
            if self.wake.wait(max(0.0, self.interval - (time.time() - t0))):
                self.wake.clear()


def _grab_screen() -> np.ndarray:
    from PIL import ImageGrab
    img = ImageGrab.grab()
    return np.asarray(img)[:, :, ::-1].copy()     # RGB -> BGR
