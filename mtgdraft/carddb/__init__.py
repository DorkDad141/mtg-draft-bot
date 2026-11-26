"""Card database: maps client card IDs (Arena grpId, later MTGO IDs) to card names.

Card names are the join key with 17Lands data and the ranking model.
"""

from .arena import ArenaCardDB, find_arena_card_db

__all__ = ["ArenaCardDB", "find_arena_card_db"]
