"""Download 17Lands public datasets for one set and event type.

Rerunnable: a file is downloaded again only when 17Lands reports a newer
last_updated date, the server's ETag changed, or the local copy is missing.

Examples:
    python scripts/download_17lands.py --set "Powered Cube" --event PremierDraft
    python scripts/download_17lands.py --set FDN --event PremierDraft --types draft game replay
    python scripts/download_17lands.py --list

Data: 17Lands public datasets, CC BY 4.0 (https://www.17lands.com/public_datasets).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

DATASETS_API = "https://www.17lands.com/api/public_datasets"
USER_AGENT = "mtg-draft-bot/0.1 (+https://github.com/DorkDad141/mtg-draft-bot)"
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "data" / "raw"
MANIFEST_NAME = "manifest.json"
CHUNK = 1 << 20


def _request(url: str, method: str = "GET") -> urllib.request.Request:
    return urllib.request.Request(url, method=method, headers={"User-Agent": USER_AGENT})


def fetch_dataset_list() -> list[dict]:
    with urllib.request.urlopen(_request(DATASETS_API), timeout=60) as resp:
        return json.load(resp)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _url_code(entry: dict) -> str:
    """Set code as it appears in file names, e.g. 'Cube_-_Powered' or 'FDN'."""
    url = next(iter(entry["urls"].values()))
    m = re.search(r"_data_public\.(.+)\.[^.]+\.csv\.gz$", url)
    return m.group(1) if m else entry["expansion"]


def find_entry(datasets: list[dict], set_name: str, event: str) -> dict:
    want = _norm(set_name)
    matches = [
        d for d in datasets
        if _norm(d["event_type"]) == _norm(event)
        and want in (_norm(d["expansion"]), _norm(_url_code(d)))
    ]
    if not matches:
        events = sorted(d["event_type"] for d in datasets if want in (_norm(d["expansion"]), _norm(_url_code(d))))
        hint = f" Event types available for that set: {', '.join(events)}." if events else " Use --list to see sets."
        sys.exit(f"No 17Lands dataset for set={set_name!r} event={event!r}.{hint}")
    return matches[0]


def load_manifest(out_dir: Path) -> dict:
    path = out_dir / MANIFEST_NAME
    return json.loads(path.read_text()) if path.exists() else {}


def save_manifest(out_dir: Path, manifest: dict) -> None:
    (out_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True))


def remote_etag(url: str) -> str | None:
    try:
        with urllib.request.urlopen(_request(url, "HEAD"), timeout=60) as resp:
            return resp.headers.get("ETag")
    except Exception:
        return None


def download(url: str, dest: Path) -> str | None:
    """Stream url to dest atomically; return the response ETag."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(_request(url), timeout=120) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        etag = resp.headers.get("ETag")
        done, start, last = 0, time.time(), 0.0
        with open(tmp, "wb") as f:
            while chunk := resp.read(CHUNK):
                f.write(chunk)
                done += len(chunk)
                now = time.time()
                if now - last > 0.5 or done == total:
                    last = now
                    pct = f"{100 * done / total:5.1f}%" if total else ""
                    rate = done / max(now - start, 1e-6) / 1e6
                    print(f"\r    {done / 1e6:8.1f} MB {pct} {rate:6.1f} MB/s", end="", flush=True)
        print()
    if total and done != total:
        tmp.unlink(missing_ok=True)
        raise IOError(f"incomplete download: got {done} of {total} bytes")
    tmp.replace(dest)
    return etag


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--set", help="Set code or name, e.g. FDN or 'Powered Cube'")
    p.add_argument("--event", default="PremierDraft", help="Event type (default PremierDraft)")
    p.add_argument("--types", nargs="+", default=["draft", "game"],
                   help="File types to fetch: draft game replay (default: draft game)")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output folder (default data/raw)")
    p.add_argument("--force", action="store_true", help="Download even if up to date")
    p.add_argument("--list", action="store_true", help="List available datasets and exit")
    args = p.parse_args()

    datasets = fetch_dataset_list()

    if args.list:
        for d in sorted(datasets, key=lambda d: (d["last_updated"], d["expansion"]), reverse=True):
            print(f"{d['last_updated']}  {d['expansion']:<16} {d['event_type']:<18} {','.join(d['urls'])}")
        return 0
    if not args.set:
        p.error("--set is required (or use --list)")

    entry = find_entry(datasets, args.set, args.event)
    print(f"{entry['expansion']} {entry['event_type']} (17Lands last_updated {entry['last_updated']})")

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(args.out)
    failures = 0

    for ftype in args.types:
        url = entry["urls"].get(ftype)
        if not url:
            print(f"  {ftype}: not published for this set, skipping")
            continue
        dest = args.out / url.rsplit("/", 1)[-1]
        prev = manifest.get(dest.name, {})
        if not args.force and dest.exists() and prev.get("last_updated") == entry["last_updated"]:
            etag = remote_etag(url)
            if etag is None or etag == prev.get("etag"):
                print(f"  {ftype}: up to date ({dest.name})")
                continue
        print(f"  {ftype}: downloading {dest.name}")
        try:
            etag = download(url, dest)
        except Exception as e:  # keep going with the other files
            print(f"  {ftype}: FAILED: {e}")
            failures += 1
            continue
        manifest[dest.name] = {
            "expansion": entry["expansion"],
            "event_type": entry["event_type"],
            "type": ftype,
            "url": url,
            "last_updated": entry["last_updated"],
            "etag": etag,
            "bytes": dest.stat().st_size,
            "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        save_manifest(args.out, manifest)

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
