"""Draft-state readers: turn a client's log into packs seen and picks made.

Each client (Arena now, MTGO later) gets its own reader that produces DraftState.
"""

from .state import DraftState
from .arena_log import ArenaLogReader, default_arena_log_path

__all__ = ["DraftState", "ArenaLogReader", "default_arena_log_path"]
