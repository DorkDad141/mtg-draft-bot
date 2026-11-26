"""MTG Arena Player.log reader.

Arena writes one line per client event. The lines used:

  [UnityCrossThreadLogger]Draft.Notify {"draftId":..,"SelfPick":p,"SelfPack":k,"PackCards":"id,id,.."}
      The pack the user is shown (Premier/Traditional human drafts). Logged even
      without "Detailed Logs".
  [UnityCrossThreadLogger]==> EventPlayerDraftMakePick {"id":..,"request":"{\"DraftId\":..,\"GrpIds\":[id],\"Pack\":k,\"Pick\":p}"}
      The user's pick. Needs "Detailed Logs (Plugin Support)" enabled in Arena.
  Any JSON line with a course object {"InternalEventName":..,"CardPool":[ids..]}
      E.g. the DraftCompleteDraft response. Gives the event name and the final pool.
  Client.SceneChange ... "Entering table draft queue: SOS_Premier_Draft"
      Event hint when no course line is present.

Pack and pick numbers are 1-based. The log is replaced each time Arena starts, and
the previous session is kept as Player-prev.log.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable, Iterator

from .state import DraftState

NOTIFY = "Draft.Notify "
MAKE_PICK = "==> EventPlayerDraftMakePick "
COMPLETE = "<== DraftCompleteDraft("
QUEUE_RE = re.compile(r"Entering table draft queue: ([A-Za-z0-9_]+)")


def default_arena_log_path() -> Path:
    """Arena's Player.log location on Windows (and macOS)."""
    win = Path(os.path.expanduser(r"~\AppData\LocalLow\Wizards Of The Coast\MTGA\Player.log"))
    mac = Path(os.path.expanduser("~/Library/Logs/Wizards Of The Coast/MTGA/Player.log"))
    return win if win.exists() or os.name == "nt" else mac


def _json_after(line: str, marker: str) -> dict | None:
    i = line.find("{", line.find(marker) + len(marker))
    if i < 0:
        return None
    try:
        return json.loads(line[i:])
    except json.JSONDecodeError:
        return None


def _find_courses(obj) -> Iterator[dict]:
    if isinstance(obj, dict):
        if "CardPool" in obj and "InternalEventName" in obj:
            yield obj
        for v in obj.values():
            yield from _find_courses(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _find_courses(v)


class ArenaLogReader:
    """Incremental parser. Feed lines (or whole files) in order; read `drafts`.

    `on_update(state, kind)` is called after each change, with kind one of
    "pack", "pick", "pool", "complete".
    """

    def __init__(self, on_update: Callable[[DraftState, str], None] | None = None, pod_size: int = 8):
        self.drafts: dict[str, DraftState] = {}
        self.on_update = on_update
        self.pod_size = pod_size
        self._last_draft: str | None = None
        self._event_hint: str | None = None
        self._expect_complete = False

    # ---- public -------------------------------------------------------------
    def feed(self, lines: Iterable[str]) -> None:
        for line in lines:
            self._line(line)

    def read_file(self, path: str | Path) -> "ArenaLogReader":
        with open(path, encoding="utf-8", errors="replace") as f:
            self.feed(f)
        self.finalize()
        return self

    def finalize(self) -> None:
        for d in self.drafts.values():
            d.infer_missing_picks()

    def follow(self, path: str | Path, poll: float = 0.25, stop: Callable[[], bool] = lambda: False) -> None:
        """Tail a live Player.log, handling Arena restarts (file replaced or truncated)."""
        path = Path(path)
        pos, ident = 0, None
        buf = ""
        while not stop():
            try:
                st = path.stat()
            except FileNotFoundError:
                time.sleep(poll)
                continue
            new_ident = (st.st_ctime_ns if os.name == "nt" else st.st_ino)
            if ident is not None and (new_ident != ident or st.st_size < pos):
                pos, buf = 0, ""            # Arena restarted: new log
            ident = new_ident
            if st.st_size > pos:
                with open(path, encoding="utf-8", errors="replace", newline="") as f:
                    f.seek(pos)
                    chunk = f.read()
                    pos = f.tell()
                buf += chunk
                *lines, buf = buf.split("\n")
                self.feed(lines)
            else:
                time.sleep(poll)

    def latest(self) -> DraftState | None:
        return self.drafts.get(self._last_draft) if self._last_draft else None

    # ---- parsing ------------------------------------------------------------
    def _draft(self, draft_id: str) -> DraftState:
        d = self.drafts.get(draft_id)
        if d is None:
            d = self.drafts[draft_id] = DraftState(draft_id, event_name=self._event_hint, pod_size=self.pod_size)
        self._last_draft = draft_id
        return d

    def _emit(self, d: DraftState, kind: str) -> None:
        if self.on_update:
            self.on_update(d, kind)

    def _line(self, line: str) -> None:
        if NOTIFY in line:
            j = _json_after(line, NOTIFY)
            if j and "draftId" in j:
                d = self._draft(j["draftId"])
                cards = [int(x) for x in str(j.get("PackCards", "")).split(",") if x.strip()]
                d.packs[(int(j["SelfPack"]), int(j["SelfPick"]))] = cards
                self._emit(d, "pack")
            return
        if MAKE_PICK in line:
            j = _json_after(line, MAKE_PICK)
            try:
                req = json.loads(j["request"]) if j else None
            except (json.JSONDecodeError, TypeError, KeyError):
                req = None
            if req and "DraftId" in req and req.get("GrpIds"):
                d = self._draft(req["DraftId"])
                d.picks[(int(req["Pack"]), int(req["Pick"]))] = int(req["GrpIds"][0])
                self._emit(d, "pick")
            return
        if COMPLETE in line:
            self._expect_complete = True
            return
        if "Entering table draft queue" in line:
            m = QUEUE_RE.search(line)
            if m:
                self._event_hint = m.group(1)
            return
        if '"CardPool"' in line:
            j = _json_after(line, "")
            if j is not None:
                for course in _find_courses(j):
                    self._course(course)
            self._expect_complete = False

    def _course(self, course: dict) -> None:
        pool = [int(x) for x in course.get("CardPool") or []]
        event = course.get("InternalEventName")
        if not pool:
            return
        # Link the pool to the draft whose logged picks it contains (best overlap).
        best, best_score = None, 0
        pc = Counter(pool)
        for d in self.drafts.values():
            known = Counter(d.picks.values())
            seen = {c for cards in d.packs.values() for c in cards}
            if known and known - pc:
                continue                      # a logged pick isn't in this pool
            score = sum(known.values()) * 10 + len(seen & set(pool))
            if score > best_score:
                best, best_score = d, score
        if best is None:
            return
        best.final_pool = pool
        if event:
            best.event_name = event
        if self._expect_complete or "DeckSelect" in str(course.get("CurrentModule", "")):
            best.complete = True
        self._emit(best, "pool")


def set_and_event(d: DraftState, set_of: Callable[[int], str] | None = None) -> tuple[str | None, str]:
    """Guess (set code, 17Lands event type) for a draft.

    Uses the event name ('PremierDraft_SOS_20260421' or the queue hint 'SOS_Premier_Draft'),
    falling back to the most common set code among the cards seen.
    """
    name = d.event_name or ""
    event = "PremierDraft"
    for key, ev in (("Premier", "PremierDraft"), ("Trad", "TradDraft"), ("Quick", "QuickDraft"),
                    ("PickTwo", "PickTwoDraft")):
        if key.lower() in name.lower():
            event = ev
            break
    m = re.match(r"^[A-Za-z]+Draft_([A-Z0-9]+)_\d+", name) or re.match(r"^([A-Z0-9]{3,4})_", name)
    set_code = m.group(1) if m else None
    if set_code is None and set_of is not None:
        counts = Counter(set_of(c) for cards in d.packs.values() for c in cards)
        set_code = counts.most_common(1)[0][0] if counts else None
    return set_code, event
