"""Keep every Arena session's log.

Arena keeps only two logs: Player.log (current session) and Player-prev.log (the one
before). Each launch overwrites Player-prev.log, so older sessions are lost. This
copies each session into an archive folder as a gzip file named after the session's
start time, and updates the copy as a running session's log grows.

A session is identified by the first client timestamp in its log (for example
"11/26/2026 10:50:18 AM"), falling back to a hash of the log's first 64 KB.
"""

from __future__ import annotations

import gzip
import hashlib
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .arena_log import default_arena_log_path

STAMP = re.compile(rb"(\d{1,2})/(\d{1,2})/(\d{4}) (\d{1,2}):(\d{2}):(\d{2}) ?([AP]M)?")
HEAD_BYTES = 512 * 1024


@dataclass
class ArchiveResult:
    source: Path
    target: Path | None
    action: str          # "new", "updated", "unchanged", "missing", "empty"


def session_key(path: Path) -> str:
    with open(path, "rb") as f:
        head = f.read(HEAD_BYTES)
    m = STAMP.search(head)
    if m:
        mo, d, y, h, mi, s, ampm = m.groups()
        hour = int(h)
        if ampm == b"PM" and hour < 12:
            hour += 12
        if ampm == b"AM" and hour == 12:
            hour = 0
        try:
            return datetime(int(y), int(mo), int(d), hour, int(mi), int(s)).strftime("%Y-%m-%d_%H%M%S")
        except ValueError:
            pass
    return "hash_" + hashlib.sha1(head[:65536]).hexdigest()[:12]


def has_drafts(path: Path) -> bool:
    with open(path, "rb") as f:
        for line in f:
            if b"Draft.Notify" in line or b"EventPlayerDraftMakePick" in line or b"BotDraft" in line:
                return True
    return False


def archive_logs(arena_dir: Path | None = None, archive_dir: Path | None = None,
                 drafts_only: bool = False) -> list[ArchiveResult]:
    arena_dir = Path(arena_dir) if arena_dir else default_arena_log_path().parent
    archive_dir = Path(archive_dir) if archive_dir else Path(__file__).resolve().parents[2] / "logs" / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for name in ("Player-prev.log", "Player.log"):
        src = arena_dir / name
        if not src.exists():
            results.append(ArchiveResult(src, None, "missing"))
            continue
        size = src.stat().st_size
        if size == 0 or (drafts_only and not has_drafts(src)):
            results.append(ArchiveResult(src, None, "empty"))
            continue
        target = archive_dir / f"Player_{session_key(src)}.log.gz"
        meta = target.with_suffix(".size")          # original size of what's archived
        old = int(meta.read_text()) if meta.exists() else -1
        if target.exists() and old >= size:
            results.append(ArchiveResult(src, target, "unchanged"))
            continue
        tmp = target.with_suffix(".tmp")
        with open(src, "rb") as fin, gzip.open(tmp, "wb", compresslevel=6) as fout:
            shutil.copyfileobj(fin, fout, 1 << 20)
        tmp.replace(target)
        meta.write_text(str(size))
        results.append(ArchiveResult(src, target, "updated" if old >= 0 else "new"))
    return results


def open_log(path: str | Path):
    """Open a Player.log or an archived Player_*.log.gz as text."""
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, encoding="utf-8", errors="replace")
