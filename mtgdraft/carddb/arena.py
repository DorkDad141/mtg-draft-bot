"""Arena card database, read from the client's own local SQLite file.

Arena ships Raw_CardDatabase_<hash>.mtga (SQLite) in MTGA_Data/Downloads/Raw/.
It is opened read-only. The file name hash changes when Arena updates, so the
newest matching file is used.
"""

from __future__ import annotations

import glob
import os
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ARENA_DIRS = [
    r"C:\Program Files\Wizards of the Coast\MTGA\MTGA_Data\Downloads\Raw",
    r"C:\Program Files (x86)\Wizards of the Coast\MTGA\MTGA_Data\Downloads\Raw",
    r"C:\Program Files\Epic Games\MagicTheGathering\MTGA_Data\Downloads\Raw",
    os.path.expanduser(r"~\AppData\LocalLow\Wizards Of The Coast\MTGA\Downloads\Raw"),
    os.path.expanduser("~/Library/Application Support/com.wizards.mtga/Downloads/Raw"),
]

RARITY = {0: "token", 1: "basic", 2: "common", 3: "uncommon", 4: "rare", 5: "mythic"}
COLOR = {"1": "W", "2": "U", "3": "B", "4": "R", "5": "G"}
CARD_TYPE = {"1": "Artifact", "2": "Creature", "3": "Enchantment", "4": "Instant", "5": "Land",
             "8": "Planeswalker", "10": "Sorcery", "11": "Kindred", "14": "Battle"}
_TAG = re.compile(r"<[^>]+>")


def find_arena_card_db(extra_dirs: list[str] | None = None) -> Path:
    for d in (extra_dirs or []) + DEFAULT_ARENA_DIRS:
        files = glob.glob(os.path.join(d, "Raw_CardDatabase_*.mtga"))
        if files:
            return Path(max(files, key=os.path.getmtime))
    raise FileNotFoundError("Arena card database (Raw_CardDatabase_*.mtga) not found. "
                            "Pass its folder explicitly.")


@dataclass(frozen=True)
class ArenaCard:
    grp_id: int
    name: str
    set_code: str
    rarity: str
    collector_number: str
    colors: str = ""            # e.g. "RW", in WUBRG order; "" for colorless
    color_identity: str = ""    # includes lands' produced colors
    types: tuple = ()           # e.g. ("Creature",)
    mana_cost: str = ""         # e.g. "{1}{U}"
    mana_value: int = 0


def _colors(field: str) -> str:
    got = {COLOR[c] for c in (field or "").split(",") if c in COLOR}
    return "".join(c for c in "WUBRG" if c in got)


def _parse_mana(text: str) -> tuple[str, int]:
    """Arena's 'o2oRoW' -> ('{2}{R}{W}', 4). Hybrid/phyrexian like 'o(R/W)' count as 1,
    and twobrid 'o(2/G)' counts as 2."""
    symbols = [t for t in text.split("o") if t]
    mv = 0
    for t in symbols:
        t = t.strip("()")
        first = t.split("/")[0]
        if first.isdigit():
            mv += int(first)
        elif t not in ("X", "Y", "Z"):
            mv += 1
    return "".join("{" + t.strip("()") + "}" for t in symbols), mv


class ArenaCardDB:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else find_arena_card_db()
        con = sqlite3.connect(f"file:{self.path.as_posix()}?mode=ro", uri=True)
        try:
            # Formatted=0 is plain text; 1 can contain markup like <nobr>. Prefer the lowest.
            rows = con.execute(
                """
                SELECT c.GrpId, l.Loc, l.Formatted, c.ExpansionCode, c.Rarity, c.CollectorNumber,
                       c.Colors, c.ColorIdentity, c.Types, c.OldSchoolManaText
                FROM Cards c JOIN Localizations_enUS l ON l.LocId = c.TitleId
                ORDER BY c.GrpId, l.Formatted
                """).fetchall()
        finally:
            con.close()
        self._cards: dict[int, ArenaCard] = {}
        for grp, loc, _fmt, exp, rar, cn, colors, ident, types, mana in rows:
            if grp in self._cards or loc is None:
                continue
            cost, mv = _parse_mana(mana or "")
            self._cards[grp] = ArenaCard(
                grp, _TAG.sub("", loc).strip(), exp or "", RARITY.get(rar, str(rar)), str(cn or ""),
                colors=_colors(colors), color_identity=_colors(ident),
                types=tuple(CARD_TYPE.get(t, t) for t in (types or "").split(",") if t),
                mana_cost=cost, mana_value=mv)

    def __len__(self) -> int:
        return len(self._cards)

    def get(self, grp_id: int) -> ArenaCard | None:
        return self._cards.get(int(grp_id))

    def name(self, grp_id: int) -> str:
        c = self._cards.get(int(grp_id))
        return c.name if c else f"<unknown grpId {grp_id}>"

    def names(self, grp_ids) -> list[str]:
        return [self.name(g) for g in grp_ids]

    def grp_for_name(self, name: str, set_code: str | None = None) -> int | None:
        """An Arena grpId for a card name, preferring a printing from `set_code`."""
        if not hasattr(self, "_by_name_all"):
            self._by_name_all: dict[str, list[ArenaCard]] = {}
            for c in self._cards.values():
                if c.rarity != "token":
                    self._by_name_all.setdefault(c.name, []).append(c)
        cands = self._by_name_all.get(name)
        if not cands:
            return None
        if set_code:
            in_set = [c for c in cands if c.set_code == set_code]
            if in_set:
                return min(in_set, key=lambda c: c.grp_id).grp_id
        return max(cands, key=lambda c: c.grp_id).grp_id

    def by_name(self) -> dict[str, ArenaCard]:
        """One card record per name (non-token printings), for color/type lookups by name."""
        out: dict[str, ArenaCard] = {}
        for c in self._cards.values():
            if c.rarity != "token" and c.name not in out:
                out[c.name] = c
        return out
