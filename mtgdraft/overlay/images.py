"""Card art from Scryfall, cached on disk, used to find cards on screen.

Lookup order for a card: Scryfall by Arena ID (/cards/arena/{grpId}), then by exact
name within the set (/cards/named?exact=..&set=..), then by exact name alone. New
sets often have no Arena IDs on Scryfall yet, so the name lookups matter.

Images go to data/card_images/<set>/<grpId>.<kind>.jpg. Scryfall asks clients to
send a User-Agent and Accept header and to keep to ~10 requests a second.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / "data" / "card_images"
HEADERS = {"User-Agent": "mtg-draft-bot/0.1", "Accept": "application/json"}
API = "https://api.scryfall.com"


class CardImageCache:
    def __init__(self, cache_dir: Path = CACHE_DIR, min_interval: float = 0.12):
        self.cache_dir = Path(cache_dir)
        self.min_interval = min_interval
        self._last = 0.0
        self._lock = threading.Lock()

    def _get(self, url: str, binary: bool = False):
        with self._lock:
            wait = self.min_interval - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        return data if binary else json.loads(data)

    def _card_json(self, grp_id: int, name: str, set_code: str | None) -> dict | None:
        tries = [f"{API}/cards/arena/{grp_id}"]
        q = urllib.parse.quote(name.split(" // ")[0])
        if set_code:
            tries.append(f"{API}/cards/named?exact={q}&set={set_code.lower()}")
        tries.append(f"{API}/cards/named?exact={q}")
        for url in tries:
            try:
                return self._get(url)
            except urllib.error.HTTPError as e:
                if e.code != 404:
                    raise
        return None

    def path(self, grp_id: int, set_code: str | None, kind: str = "art_crop") -> Path:
        return self.cache_dir / (set_code or "misc") / f"{grp_id}.{kind}.jpg"

    def get(self, grp_id: int, name: str, set_code: str | None, kind: str = "art_crop") -> Path | None:
        """Local path of the card's image (downloading it if needed), or None if not found."""
        p = self.path(grp_id, set_code, kind)
        if p.exists():
            return p
        miss = p.with_suffix(".missing")
        if miss.exists():
            return None
        try:
            card = self._card_json(grp_id, name, set_code)
        except (urllib.error.URLError, TimeoutError, OSError):
            return None                      # offline: try again next time
        uris = None
        if card:
            uris = card.get("image_uris") or (card.get("card_faces") or [{}])[0].get("image_uris")
        p.parent.mkdir(parents=True, exist_ok=True)
        if not uris or kind not in uris:
            miss.write_text(name)
            return None
        try:
            data = self._get(uris[kind], binary=True)
        except (urllib.error.URLError, TimeoutError, OSError):
            return None
        tmp = p.with_suffix(".part")
        tmp.write_bytes(data)
        tmp.replace(p)
        return p

    def prefetch(self, cards: list[tuple[int, str, str | None]], kind: str = "art_crop",
                 progress=None) -> int:
        """Download images for many cards (e.g. a whole set). Returns how many are available."""
        n = 0
        for i, (g, name, s) in enumerate(cards):
            if self.get(g, name, s, kind):
                n += 1
            if progress and (i + 1) % 25 == 0:
                progress(i + 1, len(cards))
        return n
