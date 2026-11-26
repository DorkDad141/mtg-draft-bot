"""Offline test of the on-screen card locator with synthetic draft screens.

Builds fake "pack" screenshots from real card images (Scryfall 'normal' images,
which include the frame, like Arena shows) laid out in a grid at several
resolutions and card sizes, with blur and JPEG noise, then checks that the
locator finds every card and puts its box in the right place.

    python scripts/test_locator.py --set TRK
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mtgdraft.carddb import ArenaCardDB  # noqa: E402
from mtgdraft.overlay.images import CardImageCache  # noqa: E402
from mtgdraft.overlay.locate import CardLocator, _box_iou  # noqa: E402


def make_screen(cards, normal_paths, screen_w, screen_h, card_w, cols, seed):
    rng = random.Random(seed)
    bg = np.full((screen_h, screen_w, 3), (28, 22, 18), np.uint8)
    cv2.rectangle(bg, (0, 0), (screen_w, int(0.08 * screen_h)), (50, 40, 30), -1)   # fake UI bar
    card_h = int(card_w * 1.3968)
    gap = int(card_w * 0.08)
    x_start = (screen_w - (cols * card_w + (cols - 1) * gap)) // 2
    y_start = int(0.15 * screen_h)
    truth = []
    for i, key in enumerate(cards):
        r, c = divmod(i, cols)
        img = cv2.imread(str(normal_paths[key]))
        img = cv2.resize(img, (card_w, card_h), interpolation=cv2.INTER_AREA)
        x, y = x_start + c * (card_w + gap), y_start + r * (card_h + gap)
        if y + card_h > screen_h:
            continue
        bg[y:y + card_h, x:x + card_w] = img
        truth.append((key, (x, y, x + card_w, y + card_h)))
    bg = cv2.GaussianBlur(bg, (3, 3), 0.6)
    ok, enc = cv2.imencode(".jpg", bg, [cv2.IMWRITE_JPEG_QUALITY, rng.choice([70, 85, 95])])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR), truth


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default="TRK")
    ap.add_argument("--packs", type=int, default=6)
    ap.add_argument("--save", type=Path, help="Save the annotated first screen here")
    args = ap.parse_args()

    db = ArenaCardDB()
    cache = CardImageCache()
    pool = sorted({c.name: c for c in db._cards.values() if c.set_code == args.set and c.rarity not in ("token", "basic")}.values(),
                  key=lambda c: c.grp_id)
    rng = random.Random(1)
    locator = CardLocator()
    layouts = [(1920, 1080, 190, 7), (2560, 1440, 250, 7), (1366, 768, 120, 7), (3840, 2160, 360, 7),
               (1920, 1080, 230, 5), (1280, 720, 115, 7)]
    tot = found_ok = 0
    times = []
    for p in range(args.packs):
        sw, sh, cw, cols = layouts[p % len(layouts)]
        pack = rng.sample(pool, 14)
        if p % 2:
            pack[5] = pack[4]                      # a duplicate copy in the pack
        keys = [c.grp_id for c in pack]
        normal = {c.grp_id: cache.get(c.grp_id, c.name, args.set, "normal") for c in pack}
        art = {c.grp_id: cache.get(c.grp_id, c.name, args.set, "art_crop") for c in pack}
        screen, truth = make_screen(keys, normal, sw, sh, cw, cols, seed=p)
        counts = {k: keys.count(k) for k in set(keys)}
        t = time.time()
        res = locator.locate(screen, [(k, str(art[k]), n) for k, n in counts.items() if art[k]])
        times.append(time.time() - t)
        hits = 0
        for key, box in truth:
            best = max((_box_iou(box, f.card_box) for f in res if f.key == key), default=0)
            hits += best >= 0.7
        tot += len(truth)
        found_ok += hits
        wrong = sum(1 for f in res if max((_box_iou(f.card_box, b) for k, b in truth if k == f.key), default=0) < 0.5)
        print(f"  {sw}x{sh} card {cw}px: {hits}/{len(truth)} cards boxed (IoU>=0.7), {wrong} wrong boxes, "
              f"{times[-1] * 1000:.0f} ms")
        if args.save and p == 0:
            out = screen.copy()
            for f in res:
                cv2.rectangle(out, f.card_box[:2], f.card_box[2:], (0, 255, 0), 3)
                cv2.rectangle(out, f.art_box[:2], f.art_box[2:], (0, 200, 255), 2)
            cv2.imwrite(str(args.save), out)
    print(f"Total: {found_ok}/{tot} cards located ({found_ok / tot:.0%}); median {np.median(times) * 1000:.0f} ms per screen")
    return 0 if found_ok / tot >= 0.9 else 1


if __name__ == "__main__":
    sys.exit(main())
