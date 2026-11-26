# MTG Draft Bot: Project Notes

## Goal
A draft assistant for MTG Arena Premier Draft. It reads the current pack and your picks
from Player.log, ranks the pack by how much each card raises the win rate of the deck
you are building, and draws an overlay that blacks out everything except the top N cards
(N is chosen at launch). It should later be extendable to MTGO with minimal rework.
Constraint: use as few paid drafts as possible.

## Decisions so far
- Platform: Windows, Arena set to fullscreen, single 1920x1080 monitor (DISPLAY1).
  Windows display scaling is still unknown.
- Portability: the program must run on PCs with different resolutions and scaling. The
  overlay finds cards on screen itself (computer vision), with no hard-coded pixel positions.
- Format: Premier Draft only.
- Test logs: you have Player.log files from last season's Powered Cube drafts. They use the
  same draft log events as Premier, so they are good for testing the parser.
- 17Lands DOES publish Powered Cube datasets: PremierDraft and TradDraft, each with
  draft, game and replay data, last updated 2025-11-23. So your cube logs can also be
  checked against a model trained on matching cube data.
- Objective: maximize win rate, with cards valued in the context of the deck, not in isolation.
  Must model synergy: which cards appear together in winning decks and not in losing ones.
- Data: use the 17Lands public datasets. Use whatever data exists for a new set, with no
  separate fallback. The download/train script must be easy to rerun as data grows.
- Language: Python. You may edit the code. Training runs on a local CPU.
- Code hosting: GitHub. Repository not chosen yet.
- Architecture: four modules, so MTGO can be added later by writing a new reader and
  overlay layout:
  1. Draft-state reader
  2. Card database (Scryfall IDs link Arena and MTGO card IDs)
  3. Ranking model
  4. Overlay

## Deferred questions (revisit when relevant)
- Overlay style: solid blackout or dim? Show scores on visible cards? What happens when the
  pack has fewer than N cards?
- Deck building: should the bot also suggest a build after the draft?
- Other trackers: do you run 17Lands or Untapped alongside? Can they share the log?
- Terms-of-service comfort with drawing over the Arena client.
- Is Arena true exclusive fullscreen or borderless? Test with an overlay that only displays
  something. Switch Arena to borderless or windowed if needed.
- Is "Detailed Logs (Plugin Support)" enabled in Arena? It is required for draft data.
- Do you have existing Player.log or Player-prev.log files? Search your PC once it is
  connected via Remote Control.

## Known gotchas
- Premier/Traditional P1P1 may not appear in the log until after P1P2. Needs a workaround,
  such as a screenshot or reading a different event.

## 17Lands data access (verified 2026-11-26)
- The dataset list is JSON at https://www.17lands.com/api/public_datasets
  (the /public_datasets page is rendered by JavaScript).
- Files are served from data.17landscapes.com/analysis_data/{draft_data,game_data,replay_data}/
  {type}_data_public.{SET}.{EventType}.csv.gz
- The data is released under CC BY 4.0, so credit 17Lands.
- Not every set has every file type. Some have only draft data, some only game data.
