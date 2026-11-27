"""Screen capture for the card locator.

Live drafts capture Arena's own window (PrintWindow with PW_RENDERFULLCONTENT), which
returns the game's pixels even when other windows, such as our blackout overlay, are on
top of it. That lets the overlay stay visible: windows excluded from capture
(SetWindowDisplayAffinity) turned out to be hidden on screen too when they sit over
Arena in borderless fullscreen (seen live, 2026-11-27).

The result is placed on a screen-sized canvas at the window's client position, so
card boxes found in it are already in screen coordinates.
"""

from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes

import numpy as np

ARENA_TITLE = "MTGA"
_V = ctypes.c_void_p


class _BMI(ctypes.Structure):
    _fields_ = [("biSize", ctypes.c_uint32), ("biWidth", ctypes.c_int32), ("biHeight", ctypes.c_int32),
                ("biPlanes", ctypes.c_uint16), ("biBitCount", ctypes.c_uint16), ("biCompression", ctypes.c_uint32),
                ("biSizeImage", ctypes.c_uint32), ("biXPelsPerMeter", ctypes.c_int32),
                ("biYPelsPerMeter", ctypes.c_int32), ("biClrUsed", ctypes.c_uint32), ("biClrImportant", ctypes.c_uint32)]


def _api():
    u, g = ctypes.windll.user32, ctypes.windll.gdi32
    u.FindWindowW.restype = _V
    u.FindWindowW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR]
    u.GetClientRect.argtypes = [_V, ctypes.POINTER(wintypes.RECT)]
    u.ClientToScreen.argtypes = [_V, ctypes.POINTER(wintypes.POINT)]
    u.IsIconic.argtypes = [_V]
    u.GetWindowDC.restype = _V
    u.GetWindowDC.argtypes = [_V]
    u.ReleaseDC.argtypes = [_V, _V]
    u.PrintWindow.argtypes = [_V, _V, ctypes.c_uint]
    g.CreateCompatibleDC.restype = _V
    g.CreateCompatibleDC.argtypes = [_V]
    g.CreateCompatibleBitmap.restype = _V
    g.CreateCompatibleBitmap.argtypes = [_V, ctypes.c_int, ctypes.c_int]
    g.SelectObject.restype = _V
    g.SelectObject.argtypes = [_V, _V]
    g.GetDIBits.argtypes = [_V, _V, ctypes.c_uint, ctypes.c_uint, _V, _V, ctypes.c_uint]
    g.DeleteObject.argtypes = [_V]
    g.DeleteDC.argtypes = [_V]
    return u, g


def grab_window(title: str = ARENA_TITLE) -> np.ndarray | None:
    """Screen-sized BGR image holding the window's client area at its screen position,
    or None if the window is missing, minimized, or captures as black."""
    if sys.platform != "win32":
        return None
    u, g = _api()
    h = u.FindWindowW(None, title)
    if not h or u.IsIconic(h):
        return None
    r = wintypes.RECT()
    u.GetClientRect(h, ctypes.byref(r))
    w, hh = r.right, r.bottom
    if w <= 0 or hh <= 0:
        return None
    pt = wintypes.POINT(0, 0)
    u.ClientToScreen(h, ctypes.byref(pt))
    hdc = u.GetWindowDC(h)
    mdc = g.CreateCompatibleDC(hdc)
    bmp = g.CreateCompatibleBitmap(hdc, w, hh)
    old = g.SelectObject(mdc, bmp)
    try:
        if not u.PrintWindow(h, mdc, 3):            # PW_CLIENTONLY | PW_RENDERFULLCONTENT
            return None
        bmi = _BMI(ctypes.sizeof(_BMI), w, -hh, 1, 32, 0, 0, 0, 0, 0, 0)
        buf = np.empty((hh, w, 4), np.uint8)
        g.GetDIBits(mdc, bmp, 0, hh, buf.ctypes.data, ctypes.byref(bmi), 0)
    finally:
        g.SelectObject(mdc, old)
        g.DeleteObject(bmp)
        g.DeleteDC(mdc)
        u.ReleaseDC(h, hdc)
    img = buf[:, :, :3]
    if img.mean() < 2:                               # black: this capture method didn't work
        return None
    sw, sh = u.GetSystemMetrics(0), u.GetSystemMetrics(1)
    out = np.zeros((max(sh, pt.y + hh), max(sw, pt.x + w), 3), np.uint8)
    x0, y0 = max(pt.x, 0), max(pt.y, 0)
    out[y0:y0 + hh - (y0 - pt.y), x0:x0 + w - (x0 - pt.x)] = img[y0 - pt.y:, x0 - pt.x:]
    return out
