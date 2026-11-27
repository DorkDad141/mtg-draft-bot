"""Record a live draft session for later training and debugging.

Each session gets its own folder, captures/<date_time>/ (git-ignored: screenshots and
logs show the Arena account name):

    session.json            settings (top N, screen size, shield, git commit)
    events.jsonl            one JSON line per pack / located frame / pick / draft end:
                            pack contents (grpIds and names), the ranking with scores,
                            projected deck, card boxes found on screen, and the pick made
    P1P01_before.jpg        the screen as Arena showed it (what the locator saw), no blackout
    P1P01_after.jpg         the same frame with the overlay drawn on it, i.e. what you saw
    Player.log.gz           the Arena log, copied when a draft completes and at exit

The overlay is excluded from screen capture, so "after" images are rendered from the
exact boxes the overlay drew, on top of the "before" frame.
"""

from __future__ import annotations

import gzip
import json
import shutil
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAPTURE_DIR = REPO_ROOT / "captures"


class SessionRecorder:
    def __init__(self, root: Path = DEFAULT_CAPTURE_DIR, log_path: Path | None = None,
                 settings: dict | None = None, jpeg_quality: int = 92, printer=print):
        self.dir = Path(root) / time.strftime("%Y-%m-%d_%H%M%S")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.log_path = Path(log_path) if log_path else None
        self.quality = jpeg_quality
        self.print = printer
        self.lock = threading.Lock()
        self._best: dict = {}          # (draft_id, slot) -> most cards found in a saved frame
        info = dict(settings or {})
        info["started"] = time.strftime("%Y-%m-%d %H:%M:%S")
        info["git_commit"] = _git_commit()
        (self.dir / "session.json").write_text(json.dumps(info, indent=2))
        self.print(f"Recording this session to {self.dir}")

    # ---- events -------------------------------------------------------------------
    def event(self, kind: str, **data) -> None:
        data = {"t": round(time.time(), 3), "kind": kind, **data}
        with self.lock, open(self.dir / "events.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(data, default=_jsonable) + "\n")

    def frame(self, draft_id: str, slot: tuple[int, int], screen_bgr: np.ndarray, found, top_keys,
              drawn: dict, advisor_rect=None, opacity: float = 1.0) -> None:
        """Save before/after images for this pick when more cards were found than before."""
        key = (draft_id, tuple(slot))
        if not found or len(found) <= self._best.get(key, 0):
            return
        self._best[key] = len(found)
        stem = f"P{slot[0]}P{slot[1]:02d}"
        try:
            import cv2
            before = screen_bgr
            after = render_overlay(before, drawn, advisor_rect, opacity)
            q = [cv2.IMWRITE_JPEG_QUALITY, self.quality]
            cv2.imwrite(str(self.dir / f"{stem}_before.jpg"), before, q)
            cv2.imwrite(str(self.dir / f"{stem}_after.jpg"), after, q)
        except Exception as e:  # recording must never break the assistant
            self.print(f"[recorder] could not save {stem} images: {e!r}")
            return
        self.event("frame", draft_id=draft_id, slot=list(slot), image=f"{stem}_before.jpg",
                   found=[{"grp": f.key, "card_box": f.card_box, "art_box": f.art_box, "inliers": f.inliers,
                           "top": f.key in top_keys} for f in found],
                   drawn=drawn, advisor_rect=advisor_rect, screen=list(screen_bgr.shape[1::-1]))

    def save_log(self) -> None:
        """Copy the Arena log (gzipped) into the session folder."""
        if not self.log_path or not self.log_path.exists():
            return
        try:
            with open(self.log_path, "rb") as src, gzip.open(self.dir / "Player.log.gz", "wb") as dst:
                shutil.copyfileobj(src, dst)
        except OSError as e:
            self.print(f"[recorder] could not copy the log: {e}")

    def close(self) -> None:
        self.save_log()
        n = len(list(self.dir.glob("*_before.jpg")))
        self.print(f"Recorded {n} picks with screenshots in {self.dir}")


def render_overlay(screen_bgr: np.ndarray, drawn: dict, advisor_rect=None, opacity: float = 1.0) -> np.ndarray:
    """Draw what the overlay showed (cover, holes, black boxes, badges) onto a screenshot."""
    import cv2
    out = screen_bgr.copy()
    h, w = out.shape[:2]
    mask = np.zeros((h, w), np.uint8)

    def clip(b):
        x0, y0, x1, y1 = (int(v) for v in b)
        return max(0, x0), max(0, y0), min(w, x1), min(h, y1)

    cover = drawn.get("cover")
    if cover is not None:
        x0, y0, x1, y1 = (0, 0, w, h) if cover == "full" else clip(cover)
        mask[y0:y1, x0:x1] = 1
    for b in drawn.get("black", []):
        x0, y0, x1, y1 = clip(b)
        mask[y0:y1, x0:x1] = 1
    for b in drawn.get("holes", []) + ([advisor_rect] if advisor_rect and cover is not None else []):
        x0, y0, x1, y1 = clip(b)
        mask[y0:y1, x0:x1] = 0
    m = mask.astype(bool)
    out[m] = (out[m] * (1.0 - opacity)).astype(np.uint8)
    for b, text in drawn.get("badges", []):
        x0, y0 = int(b[0]), int(b[1])
        bw = max(36, 9 * len(text))
        cv2.rectangle(out, (x0 + 4, y0 + 4), (x0 + 4 + bw, y0 + 26), (94, 215, 255), -1)
        cv2.putText(out, text, (x0 + 10, y0 + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (26, 26, 26), 1, cv2.LINE_AA)
    return out


def _jsonable(o):
    if isinstance(o, (set, tuple)):
        return list(o)
    if isinstance(o, np.generic):
        return o.item()
    return str(o)


def _git_commit() -> str | None:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                           text=True, timeout=5)
        return r.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None
