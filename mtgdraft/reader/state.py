"""Client-independent draft state: packs seen, picks made, final pool.

Packs and picks are keyed by (pack, pick), both 1-based. Cards are client IDs
(Arena grpId); the card database turns them into names.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

Slot = tuple[int, int]  # (pack, pick), 1-based


@dataclass
class DraftState:
    draft_id: str
    event_name: str | None = None
    pod_size: int = 8
    packs: dict[Slot, list[int]] = field(default_factory=dict)
    picks: dict[Slot, int] = field(default_factory=dict)        # logged picks
    inferred: dict[Slot, int] = field(default_factory=dict)     # reconstructed from the final pool
    final_pool: list[int] | None = None
    complete: bool = False

    # ---- queries ------------------------------------------------------------
    def slots(self) -> list[Slot]:
        return sorted(set(self.packs) | set(self.picks))

    def pick_at(self, slot: Slot) -> int | None:
        return self.picks.get(slot, self.inferred.get(slot))

    def pool_before(self, slot: Slot) -> list[int]:
        """Cards picked before this slot (logged or inferred)."""
        out = []
        for s in self.slots():
            if s >= slot:
                break
            c = self.pick_at(s)
            if c is not None:
                out.append(c)
        return out

    def current_slot(self) -> Slot | None:
        """The latest pack shown that has no pick yet, i.e. what the user is looking at now."""
        if not self.packs:
            return None
        latest = max(self.packs)
        return None if latest in self.picks else latest

    def missing_picks(self) -> list[Slot]:
        return [s for s in sorted(self.packs) if self.pick_at(s) is None]

    def summary(self) -> str:
        n_packs = len(self.packs)
        return (f"draft {self.draft_id[:8]} {self.event_name or '?'}: {n_packs} packs seen, "
                f"{len(self.picks)} picks logged, {len(self.inferred)} inferred, "
                f"final pool {'yes' if self.final_pool else 'no'}"
                f"{', complete' if self.complete else ''}")

    # ---- reconstruction -----------------------------------------------------
    def infer_missing_picks(self) -> int:
        """Fill picks missing from the log using the final pool.

        A pick slot's card must be in that pack, and (in an N-player pod) must NOT be
        in the same pack when it comes back N picks later. Solved as a small exact
        matching; only slots with a unique answer are filled. Returns slots filled.
        """
        if not self.final_pool:
            return 0
        remaining = Counter(self.final_pool) - Counter(self.picks.values())
        missing = [s for s in sorted(self.packs) if s not in self.picks]
        if not missing or not remaining:
            return 0

        def candidates(slot: Slot) -> list[int]:
            pack, pick = slot
            later = self.packs.get((pack, pick + self.pod_size))
            return [c for c in set(self.packs[slot])
                    if remaining[c] > 0 and (later is None or c not in later)]

        cands = {s: candidates(s) for s in missing}
        order = sorted(missing, key=lambda s: len(cands[s]))
        solutions: list[dict[Slot, int]] = []
        used: Counter = Counter()
        current: dict[Slot, int] = {}

        def dfs(i: int) -> None:
            if len(solutions) >= 50:
                return
            if i == len(order):
                solutions.append(dict(current))
                return
            s = order[i]
            for c in cands[s]:
                if used[c] < remaining[c]:
                    used[c] += 1
                    current[s] = c
                    dfs(i + 1)
                    used[c] -= 1
                    del current[s]

        dfs(0)
        if not solutions:
            return 0
        filled = 0
        for s in missing:
            vals = {sol[s] for sol in solutions}
            if len(vals) == 1:
                self.inferred[s] = vals.pop()
                filled += 1
        return filled
