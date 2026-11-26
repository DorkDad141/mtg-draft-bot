"""Find known cards in a screenshot by matching their art.

The log already says which cards are in the pack; the screenshot is only used to
find where they are. Each card's art (Scryfall art_crop) is matched to the screen
with SIFT features + a RANSAC homography, so it works at any resolution, window
size or display scaling, with no hard-coded positions. The art box is then
expanded to the whole card using standard card-frame proportions.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Where the art sits in the card, as fractions of card width W / height H (H = 1.3968 W).
# Modern (M15) frame: landscape art at the top. Saga frame: portrait art on the right.
FRAMES = {
    "normal": dict(left=0.0807, width=0.8386, top=0.1117, height=0.4425),
    "saga": dict(left=0.50, width=0.42, top=0.110, height=0.725),
}
CARD_ASPECT = 1.3968


@dataclass
class Found:
    key: object
    card_box: tuple[int, int, int, int]   # x0, y0, x1, y1 in screen pixels
    art_box: tuple[int, int, int, int]
    inliers: int


def _box_iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter == 0:
        return 0.0
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / area


class CardLocator:
    def __init__(self, work_width: int = 1600, template_width: int = 320, min_inliers: int = 10,
                 ratio: float = 0.75):
        self.work_width = work_width
        self.template_width = template_width
        self.min_inliers = min_inliers
        self.ratio = ratio
        self.sift = cv2.SIFT_create()
        self._templates: dict = {}
        self._matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=48))

    def _template(self, path):
        t = self._templates.get(path)
        if t is None:
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                self._templates[path] = t = (None, None, None)
                return t
            s = self.template_width / img.shape[1]
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            kp, des = self.sift.detectAndCompute(img, None)
            t = self._templates[path] = (kp, des, img.shape)
        return t

    def locate(self, screen_bgr: np.ndarray, cards: list[tuple[object, str, int]]) -> list[Found]:
        """cards: (key, art image path, copies in pack). Returns boxes in screen pixels."""
        h, w = screen_bgr.shape[:2]
        scale = self.work_width / w          # downscale big screens, upscale small ones
        gray = cv2.cvtColor(screen_bgr, cv2.COLOR_BGR2GRAY)
        if abs(scale - 1.0) > 0.05:
            interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=interp)
        else:
            scale = 1.0
        kp_s, des_s = self.sift.detectAndCompute(gray, None)
        if des_s is None or len(kp_s) < self.min_inliers:
            return []
        pts_s = np.float32([k.pt for k in kp_s])
        found: list[Found] = []
        for key, path, copies in cards:
            kp_t, des_t, shape = self._template(path)
            if des_t is None or len(kp_t) < self.min_inliers:
                continue
            knn = self._matcher.knnMatch(des_t, des_s, k=2)
            good = [m[0] for m in knn if len(m) == 2 and m[0].distance < self.ratio * m[1].distance]
            used = np.zeros(len(kp_s), bool)
            for _ in range(max(1, copies)):
                cand = [m for m in good if not used[m.trainIdx]]
                if len(cand) < self.min_inliers:
                    break
                src = np.float32([kp_t[m.queryIdx].pt for m in cand]).reshape(-1, 1, 2)
                dst = np.float32([kp_s[m.trainIdx].pt for m in cand]).reshape(-1, 1, 2)
                H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
                if H is None or mask.sum() < self.min_inliers:
                    break
                th, tw = shape
                corners = cv2.perspectiveTransform(np.float32([[0, 0], [tw, 0], [tw, th], [0, th]]).reshape(-1, 1, 2), H)
                c = corners.reshape(-1, 2) / scale
                x0, y0 = c[:, 0].min(), c[:, 1].min()
                x1, y1 = c[:, 0].max(), c[:, 1].max()
                aw, ah = x1 - x0, y1 - y0
                # Sanity: a roughly axis-aligned, sensibly sized art box with the template's aspect.
                if aw < 0.02 * w or ah < 0.01 * h or not (0.6 < (aw / ah) / (tw / th) < 1.6) \
                        or not cv2.isContourConvex(corners.astype(np.int32)):
                    break
                fr = FRAMES["saga" if th > 1.2 * tw else "normal"]
                W = aw / fr["width"]
                Hc = W * CARD_ASPECT
                cx0, cy0 = x0 - fr["left"] * W, y0 - fr["top"] * Hc
                box = (int(max(0, cx0)), int(max(0, cy0)), int(min(w, cx0 + W)), int(min(h, cy0 + Hc)))
                found.append(Found(key, box, (int(x0), int(y0), int(x1), int(y1)), int(mask.sum())))
                # Mask screen keypoints inside this art so a second copy can be found.
                inside = ((pts_s[:, 0] >= x0 * scale) & (pts_s[:, 0] <= x1 * scale) &
                          (pts_s[:, 1] >= y0 * scale) & (pts_s[:, 1] <= y1 * scale))
                used |= inside
        # Resolve overlaps between different cards: keep the stronger match.
        found.sort(key=lambda f: -f.inliers)
        kept: list[Found] = []
        for f in found:
            if all(_box_iou(f.art_box, k.art_box) < 0.3 for k in kept):
                kept.append(f)
        return kept
