"""Overlay windows (Windows, tkinter).

* Advisor: a small always-on-top panel with the projected deck and the pack ranked,
  top N highlighted. Works in any Arena window mode.
* Blackout: a borderless, click-through, always-on-top window covering the screen.
  It is fully transparent except for black boxes over the pack cards that are NOT in
  the top N, plus a small rank badge on each top card. It is excluded from screen
  capture, so it never hides cards from the locator's screenshots.

Arena must run in windowed or borderless-windowed mode for any window to draw over
it; true exclusive fullscreen hides other windows.

All methods are thread-safe: other threads post updates, the Tk main loop applies them.
"""

from __future__ import annotations

import ctypes
import queue
import sys
import tkinter as tk
from tkinter import font as tkfont

KEY = "#010203"            # transparent color key for the blackout window
GWL_EXSTYLE = -20
WS_EX_LAYERED, WS_EX_TRANSPARENT = 0x00080000, 0x00000020
WS_EX_TOOLWINDOW, WS_EX_NOACTIVATE, WS_EX_TOPMOST = 0x00000080, 0x08000000, 0x00000008
WDA_EXCLUDEFROMCAPTURE = 0x00000011


def make_dpi_aware() -> None:
    """Use physical pixels everywhere, so screenshots and window coordinates agree."""
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except (AttributeError, OSError):
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            pass


def _hwnd(win: tk.Misc) -> int:
    win.update_idletasks()
    return ctypes.windll.user32.GetParent(win.winfo_id()) or win.winfo_id()


class OverlayUI:
    def __init__(self, top_n: int, blackout: bool = True, opacity: float = 1.0,
                 exclude_from_capture: bool = True, title: str = "Draft assistant"):
        self.top_n = top_n
        self.q: queue.Queue = queue.Queue()
        self.root = tk.Tk()
        self.root.title(title)
        self.root.attributes("-topmost", True)
        self.root.configure(bg="#1c1c20")
        sw = self.root.winfo_screenwidth()
        self.root.geometry(f"380x560+{sw - 400}+40")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.closed = False
        f = tkfont.nametofont("TkDefaultFont").actual()["family"]
        self.f_title = (f, 12, "bold")
        self.f_body = (f, 10)
        self.f_small = (f, 9)
        self.header = tk.Label(self.root, text="Waiting for a draft...", font=self.f_title, fg="#f2f2f2",
                               bg="#1c1c20", anchor="w", justify="left")
        self.header.pack(fill="x", padx=10, pady=(10, 2))
        self.lanes = tk.Label(self.root, text="", font=self.f_body, fg="#9fd4a8", bg="#1c1c20",
                              anchor="w", justify="left", wraplength=360)
        self.lanes.pack(fill="x", padx=10)
        self.listbox = tk.Text(self.root, font=self.f_body, bg="#232328", fg="#e6e6e6", relief="flat",
                               height=20, wrap="none", cursor="arrow")
        self.listbox.pack(fill="both", expand=True, padx=10, pady=6)
        self.listbox.tag_configure("top", foreground="#ffd75e", font=(f, 10, "bold"))
        self.listbox.tag_configure("dim", foreground="#8a8a90")
        self.status = tk.Label(self.root, text="", font=self.f_small, fg="#9a9aa2", bg="#1c1c20",
                               anchor="w", justify="left", wraplength=360)
        self.status.pack(fill="x", padx=10, pady=(0, 8))

        self.overlay = None
        if blackout:
            ov = tk.Toplevel(self.root)
            ov.overrideredirect(True)
            ov.geometry(f"{sw}x{self.root.winfo_screenheight()}+0+0")
            ov.configure(bg=KEY)
            ov.attributes("-transparentcolor", KEY)
            ov.attributes("-topmost", True)
            if opacity < 1.0:
                ov.attributes("-alpha", opacity)
            self.canvas = tk.Canvas(ov, bg=KEY, highlightthickness=0)
            self.canvas.pack(fill="both", expand=True)
            self.overlay = ov
            if sys.platform == "win32":
                h = _hwnd(ov)
                u = ctypes.windll.user32
                style = u.GetWindowLongW(h, GWL_EXSTYLE)
                u.SetWindowLongW(h, GWL_EXSTYLE, style | WS_EX_LAYERED | WS_EX_TRANSPARENT |
                                 WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE | WS_EX_TOPMOST)
                if exclude_from_capture:
                    u.SetWindowDisplayAffinity(h, WDA_EXCLUDEFROMCAPTURE)
        self.root.after(50, self._poll)

    # ---- thread-safe API ------------------------------------------------------
    def show_pack(self, header: str, lanes: str, rows: list[tuple[str, str, bool]], status: str = "") -> None:
        """rows: (left text, right text, is_top)."""
        self.q.put(("pack", header, lanes, rows, status))

    def set_status(self, text: str) -> None:
        self.q.put(("status", text))

    def set_boxes(self, black: list[tuple[int, int, int, int]], badges: list[tuple[tuple, str]]) -> None:
        self.q.put(("boxes", black, badges))

    def clear_boxes(self) -> None:
        self.q.put(("boxes", [], []))

    def show_fake_pack(self, image_paths: list[str]) -> None:
        """Test mode: draw the pack's card images in a 'fake Arena' window."""
        self.q.put(("fake", image_paths))

    def _do_fake(self, paths):
        from PIL import Image, ImageTk
        if not hasattr(self, "fake"):
            self.fake = tk.Toplevel(self.root)
            self.fake.title("Fake Arena (test mode)")
            self.fake.configure(bg="#16120e")
            self.fake.geometry("1500x860+30+30")
            self.fake_canvas = tk.Canvas(self.fake, bg="#16120e", highlightthickness=0)
            self.fake_canvas.pack(fill="both", expand=True)
            self.fake.attributes("-topmost", True)       # stay visible for screenshots
            self.root.lift()
            if self.overlay is not None:
                self.overlay.lift()                        # overlay above the fake screen
        c = self.fake_canvas
        c.delete("all")
        self._fake_imgs = []
        cw, gap, cols = 180, 14, 7
        ch = int(cw * 1.3968)
        for i, p in enumerate(paths):
            r, col = divmod(i, cols)
            if not p:
                continue
            im = Image.open(p).convert("RGB").resize((cw, ch))
            tkim = ImageTk.PhotoImage(im)
            self._fake_imgs.append(tkim)
            c.create_image(60 + col * (cw + gap), 90 + r * (ch + gap), image=tkim, anchor="nw")
        c.create_text(60, 40, text="Fake Arena: simulated pack (test mode)", fill="#c8b89a", anchor="w",
                      font=self.f_title)

    def close(self) -> None:
        self.closed = True
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def run(self) -> None:
        self.root.mainloop()

    # ---- Tk thread --------------------------------------------------------------
    def _poll(self) -> None:
        if self.closed:
            return
        try:
            while True:
                msg = self.q.get_nowait()
                getattr(self, "_do_" + msg[0])(*msg[1:])
        except queue.Empty:
            pass
        self.root.after(50, self._poll)

    def _do_pack(self, header, lanes, rows, status):
        self.header.config(text=header)
        self.lanes.config(text=lanes)
        lb = self.listbox
        lb.config(state="normal")
        lb.delete("1.0", "end")
        for left, right, top in rows:
            lb.insert("end", f"{left:<34}{right:>10}\n", "top" if top else ("dim" if right.strip() == "--" else ""))
        lb.config(state="disabled")
        if status:
            self.status.config(text=status)

    def _do_status(self, text):
        self.status.config(text=text)

    def _keep_on_top(self) -> None:
        """Re-assert the overlay as the topmost window without activating it."""
        if self.overlay is None or sys.platform != "win32":
            return
        from ctypes import wintypes
        u = ctypes.windll.user32
        # Declare types: on 64-bit Windows HWND_TOPMOST (-1) must be a pointer-sized handle.
        u.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int,
                                   ctypes.c_int, ctypes.c_int, ctypes.c_uint]
        u.SetWindowPos.restype = wintypes.BOOL
        SWP_NOSIZE, SWP_NOMOVE, SWP_NOACTIVATE, SWP_SHOWWINDOW = 0x1, 0x2, 0x10, 0x40
        u.SetWindowPos(_hwnd(self.overlay), wintypes.HWND(-1), 0, 0, 0, 0,
                       SWP_NOSIZE | SWP_NOMOVE | SWP_NOACTIVATE | SWP_SHOWWINDOW)

    def _do_boxes(self, black, badges):
        if self.overlay is None:
            return
        self._keep_on_top()
        c = self.canvas
        c.delete("all")
        for x0, y0, x1, y1 in black:
            c.create_rectangle(x0, y0, x1, y1, fill="black", outline="")
        for (x0, y0, x1, y1), text in badges:
            w = max(36, 9 * len(text))
            c.create_rectangle(x0 + 4, y0 + 4, x0 + 4 + w, y0 + 26, fill="#ffd75e", outline="")
            c.create_text(x0 + 4 + w / 2, y0 + 15, text=text, fill="#1a1a1a", font=self.f_small + ("bold",))
