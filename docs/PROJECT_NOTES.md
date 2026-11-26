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

## First target: Powered Cube (decided)
- Train and evaluate on 17Lands "Cube - Powered" PremierDraft first, because the user's
  logs are from that format and the whole system can be tested with no paid drafts.
  Add regular sets afterwards with the same code.
- Open question: has Arena's Powered Cube card list changed since the data's last update on
  2025-11-23? If it has, new cards will have no data yet.
- What the downloaded data looks like (checked 2026-11-26):
  - The game file (27 MB gz) has 266,118 games. Each row has columns deck_/sideboard_/
    opening_hand_/drawn_/tutored_<card name>, plus won, on_play, num_mulligans, colors,
    rank, opp_rank, user_game_win_rate_bucket and user_n_games_bucket.
  - The draft file (93 MB gz) has 2,118,674 picks. Each row has the pick, pack_card_<name>
    and pool_<name> columns, plus event_match_wins and event_match_losses.
  - There are about 620 cards. Cards are identified by name, not Arena ID, so a card
    database (Scryfall) is needed to map names to Arena IDs (grpId) and, later, MTGO IDs.

## Model plan (agreed)
- Objective: predict P(win) from a whole deck, not from cards one at a time.
- First model: a factorization machine on the game data. It learns a value for each card
  plus a pairwise synergy score for each pair of cards (through low-rank embeddings), so it
  captures cards that appear together in winning decks and not in losing ones.
- Control for player skill with user_game_win_rate_bucket (and user_n_games_bucket,
  on_play, mulligans), so cards get credit and pilots don't.
- Pick scoring: for each card in the pack, add it to the current pool, estimate the likely
  final deck (main colors, best playables), and score it with the model. Rank by the gain
  in win rate.
- Possible upgrade: a pick model trained on the draft data, predicting what strong drafters
  pick, to estimate future picks and signals. Add it only if it measurably helps.
- Evaluation: offline, on held-out 17Lands games and drafts, then on the user's own cube
  logs. No paid drafts.

## Screen and overlay plan
- The log tells us which cards are in the pack. A screenshot is only used to locate them,
  with computer vision such as matching against Scryfall card images. This must work at any
  resolution and display scaling. There are no hard-coded pixel positions.
- Early test: draw a test overlay over Arena running in fullscreen before building the full
  thing.

## Environment and setup status
- GitHub repo: DorkDad141/mtg-draft-bot.
- Cloud sessions need these network hosts allowed: www.17lands.com (dataset list API) and
  data.17landscapes.com (the files). A personal cloud environment "mtg-draft-bot" has them.
- User's PC: Windows, computer name dreammaker141. The work folder is
  %USERPROFILE%\mtg-draft-bot, which holds the Powered Cube Player.log files.
- Remote Control: the folder appears in the Remote Control list, but the PC does not offer
  it to cloud sessions. The reason isn't known; it may need a desktop app update. Fallback:
  run Claude Code locally on the PC.

## Next steps
1. `scripts/download_17lands.py`: rerunnable. Takes a set code and event type, reads
   https://www.17lands.com/api/public_datasets, downloads the matching files into
   data/raw/, and skips files whose last_updated date hasn't changed.
2. Load the game data into a sparse deck matrix and train the factorization machine. Report
   held-out log-loss and AUC against simpler baselines (per-card win rates, logistic
   regression with no pairwise terms).
3. Pick scorer and offline replay evaluation on 17Lands drafts.
4. Player.log parser, tested on the user's cube logs. Handle the P1P1 gotcha below.
5. Card database (Scryfall), then overlay.

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
