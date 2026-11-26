"""Archive Arena's Player.log / Player-prev.log so no session is lost.

Arena overwrites Player-prev.log every time it starts, keeping only the last two
sessions. Run this regularly (it is safe to run any time, even while Arena is
running) to keep a gzip copy of every session in logs/archive/ (git-ignored).

    python scripts/archive_logs.py               # archive once
    python scripts/archive_logs.py --watch 10    # keep archiving every 10 minutes
    python scripts/archive_logs.py --list        # show what is archived

To run it automatically, see "Archiving Arena logs" in the README.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mtgdraft.reader.archive import archive_logs, open_log  # noqa: E402

DEFAULT_ARCHIVE = REPO_ROOT / "logs" / "archive"


def list_archive(archive: Path) -> None:
    files = sorted(archive.glob("Player_*.log.gz"))
    if not files:
        print(f"No archived sessions in {archive}")
        return
    for f in files:
        drafts = set()
        with open_log(f) as fh:
            for line in fh:
                if '"InternalEventName":"' in line:
                    i = line.index('"InternalEventName":"') + 21
                    ev = line[i:line.index('"', i)]
                    if "Draft" in ev or "Sealed" in ev:
                        drafts.add(ev)
                elif "Entering table draft queue:" in line:
                    drafts.add(line.split("Entering table draft queue:")[1].split('"')[0].strip())
        print(f"{f.name:<36} {f.stat().st_size / 1e6:6.1f} MB gz   {', '.join(sorted(drafts)) or '-'}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arena-dir", type=Path, help="Folder with Player.log (default: Arena's LocalLow folder)")
    ap.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE, help="Archive folder (default logs/archive)")
    ap.add_argument("--drafts-only", action="store_true", help="Skip sessions with no draft activity")
    ap.add_argument("--watch", type=float, metavar="MINUTES", help="Keep running, archiving every N minutes")
    ap.add_argument("--list", action="store_true", help="List archived sessions and their draft events")
    args = ap.parse_args()

    if args.list:
        list_archive(args.archive)
        return 0
    while True:
        for r in archive_logs(args.arena_dir, args.archive, args.drafts_only):
            if r.action in ("new", "updated"):
                print(f"{time.strftime('%H:%M:%S')} {r.action:<8} {r.source.name} -> {r.target.name}", flush=True)
            elif not args.watch:
                print(f"{r.action:<9} {r.source.name}" + (f" ({r.target.name})" if r.target else ""))
        if not args.watch:
            return 0
        time.sleep(args.watch * 60)


if __name__ == "__main__":
    sys.exit(main())
