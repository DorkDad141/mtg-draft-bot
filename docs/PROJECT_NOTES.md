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
- Test logs: CORRECTED 2026-11-26. The Player.log files in repo/logs/ are NOT Powered Cube.
  They hold 6 SOS Premier Draft drafts (event PremierDraft_SOS_20260421), played on
  2026-05-09. See "Arena log findings" below. The cube model is still a useful
  benchmark, and the SOS model is used to replay these logs.
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
- No signals (2026-11-27): the picker will not read what other drafters are passing or
  estimate wheels. It values cards only by the deck being built.
- No rarity for cube (2026-11-27): rarity must not be a factor in cube card values
  (including any fallback value for cards without data).

## Deferred questions (revisit when relevant)
- Overlay style: solid blackout or dim? Show scores on visible cards? What happens when the
  pack has fewer than N cards?
- Deck building: should the bot also suggest a build after the draft?
- Other trackers: do you run 17Lands or Untapped alongside? Can they share the log?
- Terms-of-service comfort with drawing over the Arena client.
- Is Arena true exclusive fullscreen or borderless? Test with an overlay that only displays
  something. Switch Arena to borderless or windowed if needed.
- Is "Detailed Logs (Plugin Support)" enabled in Arena? It is required for the pick
  records (the packs are logged either way). The 2026-05-09 evening log has pick records,
  so it was probably on by then. Confirm before a live test.
- Which tracker caused the restart mid-draft, if any? Arena restarting mid-draft loses
  the pack notification for the pick in progress (see below).

## First target: Powered Cube (decided)
- Train and evaluate on 17Lands "Cube - Powered" PremierDraft first, because the user's
  logs are from that format and the whole system can be tested with no paid drafts.
  Add regular sets afterwards with the same code.
  (Update 2026-11-26: the logs turned out to be SOS, and SOS now runs through the same
  code, so regular sets are already supported.)
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
  %USERPROFILE%\mtg-draft-bot, which holds the Player.log files (SOS, not cube; see above).
- Remote Control: the folder appears in the Remote Control list, but the PC does not offer
  it to cloud sessions. The reason isn't known; it may need a desktop app update. Fallback:
  run Claude Code locally on the PC.
- As of 2026-11-26, work runs locally on the PC (Claude desktop app). The repo is cloned at
  %USERPROFILE%\mtg-draft-bot\repo, the Arena logs are in repo/logs/ (git-ignored), and
  Python 3.13.3 is installed. The PC can reach both 17Lands hosts.

## Next steps
1. DONE (2026-11-26): `scripts/download_17lands.py`. Takes `--set` (code like FDN or name
   like "Powered Cube"), `--event` and `--types` (default draft and game; replay is
   opt-in because it is large). Saves to data/raw/ with a manifest.json and skips files
   whose last_updated date and ETag haven't changed. Powered Cube PremierDraft draft and
   game files are downloaded locally (93 MB and 27 MB).
   Note: the API's "expansion" is the display name ("Powered Cube"), but file names use
   "Cube_-_Powered". The script accepts either.
2. DONE (2026-11-26): deck win-rate model. Code: `mtgdraft/model/` (games.py loader,
   fm.py factorization machine in numpy/scipy, metrics.py). Run
   `python scripts/train_deck_model.py --set Cube_-_Powered` (about 1 minute on CPU for the
   full grid). The best model is saved to data/models/. Results are in "Deck model results"
   below.
   Dependencies are in requirements.txt (numpy, scipy, pandas), installed in repo/.venv.
3. DONE (2026-11-26): pick scorer and offline replay evaluation. Code:
   `mtgdraft/model/scorer.py` (PickScorer; `rank(pack, pool)` for live use and
   `score_packs` for batch scoring), `drafts.py` (pick-data loader), `baselines.py`.
   Run `python scripts/eval_picks.py` (about 1 minute; the first run parses the draft
   CSV). Results are in "Pick scorer results" below. The default synergy weight is 1.0.
4. DONE (2026-11-26): Arena log reader and card database.
   - `mtgdraft/carddb/arena.py` reads Arena's own local SQLite card DB
     (MTGA_Data/Downloads/Raw/Raw_CardDatabase_*.mtga, read-only) and maps grpId to name.
     All 275 card IDs in the logs map to names that exactly match the 17Lands SOS columns.
     Scryfall is still needed later for MTGO IDs.
   - `mtgdraft/reader/` contains the client-independent DraftState (packs, picks, final
     pool, inference of missing picks) and ArenaLogReader (incremental parsing; `follow()`
     tails the live log and handles Arena restarts).
   - `python scripts/show_draft.py logs/Player-prev.log logs/Player.log` replays the logs
     through the scorer. `--live` follows the running Arena log and prints the top cards
     for each new pack (a text-only stand-in for the overlay).
   - The SOS model is trained (`train_deck_model.py --set SOS`): 582,914 games and 346
     cards. Test log-loss is 0.66092 for logistic regression and 0.65918 for the FM
     (-0.00173 +/- 0.00020, about 8.6 SE), a much clearer synergy gain than in the cube.
     The best l2_v sat at the edge of the grid (3e-3); try larger values.
   - The scorer now never recommends basic lands (they score NaN and rank last).
   - `scripts/simulate_live_log.py` replays a recorded draft into data/sim/Player.log in
     real Arena line format, while the live reader (`follow()`) ranks every pack. It is
     the way to test the live path and the future overlay without paid drafts.
     `--bot-picks` lets the bot draft packs whose picks are unknown (approximate,
     because later packs still reflect the real picks). Tested in-process and with
     `show_draft.py --live` as a separate process; both see every pack.
   - Model loading is shared through `mtgdraft.model.scorer.load_scorer(set, event)`.
5. DONE (2026-11-26): live assistant and overlay. `scripts/draft_assistant.py --top N`
   (see "Live assistant and overlay" below, and docs/TRK_TEST.md for the test plan).
6. Also done 2026-11-26: deck-aware scorer (now the default), TRK support with decks
   rebuilt from draft data, Arena log archiver, pick-by-pick HTML reports.
7. Done 2026-11-27: the shield and session recording.
   - Shield (default): the pack area turns black the moment a pick is logged, and on
     each new pack before scoring; once the cards are found, holes are cut for the top
     N only. The pack area is the union of card boxes found (6% margin), saved per
     screen size in data/overlay_region.json, so only the very first pack on a new
     screen size can show uncovered. Until the area is known, the whole screen is
     covered for at most 3 s (the advisor panel is always left visible). If no card is
     found for 3 s, the cover is lifted (Arena hidden). Rehearsal trace: 31/32 packs
     were already covered when shown (cover up about 0.45 s before the pack); the miss
     was the very first P1P1 (see the P1P1 caveat below). Holes appear about 0.8 s after the pack. `--no-shield`
     restores per-card boxes. The log is polled every 50 ms.
   - Quick/Traditional drafts fall back to the set's PremierDraft model (we only train
     on Premier Draft data).
   - Quick Draft logs: the reader also parses BotDraft_DraftStatus / BotDraft_DraftPick
     (0-based numbers, response JSON on the marker line or the next, CardIds or older
     CardId, DraftStatus "Completed" gives the pool). No draft ID, so bot drafts get
     "<EventName>#<n>". Kinds are detected per line; no setting. Written from the known
     format, verified on simulated logs (--sim-quick) only: check on the first live
     Quick Draft. Premier parsing of the archived SOS logs is unchanged.
   - P1P1 caveat: no earlier pick triggers the shield, so it rises when the pack is
     logged. Rehearsals never save the pack area (a fake window's area was saved by
     mistake once and deleted).
   - Recorder (mtgdraft/recorder.py, on by default for live runs): captures/<date_time>/
     with before/after JPEGs per pick (after = the drawn overlay rendered onto the same
     frame, because the overlay is excluded from capture), events.jsonl (packs, rankings,
     found boxes, picks, pool), session.json and Player.log.gz. These are labelled
     real Arena screens for tuning the locator, and picks for the pick model.
   - MTGDRAFT_TRACE=1 prints overlay/fake-screen timestamps for latency checks.

8. First live test (2026-11-27): TRK Quick Draft, 42 picks, recorded in
   captures/2026-11-27_13*/ (final session has all 42 packs/picks + pool; 41 of 42
   picks have before/after screenshots across sessions). Report:
   data/reports/draft_QuickDraft_TRK_20261119_1.html. Your pick = bot #1 on 40/42.
   Problems found and fixed:
   - Quick Draft messages are "BotDraftDraftStatus"/"BotDraftDraftPick" (no
     underscore); the reader now accepts both. Pick request format matched the plan
     (PickInfo.CardIds, 0-based numbers).
   - The blackout was invisible: windows excluded from capture (WDA_EXCLUDEFROMCAPTURE)
     are hidden on screen over Arena's borderless fullscreen (test: red normal box
     visible, blue excluded box not). Live, the locator now captures Arena's own window
     (PrintWindow PW_RENDERFULLCONTENT, ~50 ms, mtgdraft/overlay/capture.py) and the
     overlay is not excluded. Rehearsals still capture the full screen with exclusion.
   - The pack area grew over the deck panel: matches in the deck panel, enlarged hover
     previews, and cards caught mid slide-in. Now learned only from pack-sized boxes
     that stayed put between two checks, never past 78% of the screen width
     (data/overlay_region_v2.json; learned (247,156)-(1261,999)).
   - The overlay covered other windows: it now shows only while Arena (or the
     advisor) is the active window; the advisor is topmost only then.
   - The report tool had no Quick Draft model: load_scorer now falls back to the set's
     PremierDraft model for any event.
   Results: all top cards found on 41/41 screens; the only misses were basic lands
   (the pack's basic slot uses different art on Arena; basics stay covered, harmless).
   Pack logged -> first located frame: median 1.4 s (max 2.0): now re-checks every
   0.25 s until the top cards are found (not yet tested live). Pick logged -> next
   pack logged: median 0.26 s, so the shield rises well before the next pack shows.
   Not checked: whether P1P1 flashed before the first cover.

### Next steps (after the first live TRK test)
- Review the live test: console output, captures/<session>/ (before/after screenshots,
  events.jsonl), data/reports/ report. Check the shield never showed an unranked pack.
- If cards are missed on the real Arena screen: tune ART frame proportions in
  mtgdraft/overlay/locate.py for Arena's card rendering, or add more frame types.
- Retrain TRK when 17Lands publishes TRK game data (the pipeline switches to game
  data automatically): download_17lands.py --set TRK --types draft game, then train.
- Bot color mix vs humans (see the mana-aware section): too often 3 colors in TRK,
  too rarely splashes/3 colors in cube. And wins/agree is lower than the deck-aware
  scorer's on all three test splits (each within noise). Worth a look.
- Merge the feature branches into main.

## Deck model results (Powered Cube PremierDraft, 2026-11-26)
- Data: 266,117 games from 47,592 drafts, played 2025-10-30 to 2025-11-19, with 545 cards.
  Split by draft into 80% train, 10% validation and 10% test. Features are main-deck copy
  counts.
- Controls (linear only): skill bucket, number-of-games bucket, rank, play/draw, own and
  opponent mulligans. The skill bucket probably includes the game being predicted, which
  leaks the result for players with few games, so skill is set to "unknown" below 50
  games (SKILL_MIN_GAMES).
- Held-out test results:

  | model | log-loss | AUC |
  |---|---|---|
  | constant win rate | 0.68943 | 0.500 |
  | controls only | 0.67010 | 0.611 |
  | controls + average card win rate | 0.66673 | 0.620 |
  | logistic regression (card weights) | 0.66356 | 0.628 |
  | FM k=8, l2_v=3e-3 (card weights + synergy) | 0.66279 | 0.631 |

  FM vs logistic regression: log-loss is 0.00077 lower, with a standard error of 0.00030
  (clustered by draft), about 2.6 SE. Synergy adds a small but real signal. Validation
  results for k from 4 to 16 are close together; regularization (l2_v) matters more than k.
- Sanity check: the highest card weights are Time Walk, Ancestral Recall, Black Lotus and
  Sol Ring. The strongest synergies form a Tolarian Academy / Mox Opal / Upheaval /
  artifact cluster. The negative pairs are off-plan or off-color, such as Academy with
  non-artifact cards, Plains + Spirebluff Canal, and Island + Ragavan. Individual pair
  terms are small (about 0.01 logit), but they add up over the many pairs in a focused deck.
- Win prediction is inherently noisy (AUC around 0.63). What matters for picks is the
  ranking of cards for a given pool, which step 3 evaluates.
- Ideas if needed later: a time-based split, deck color or archetype features, and
  averaging several seeds. Shocklands, triomes and other dual lands have negative weights,
  probably because greedy multicolor decks lose more. The pick scorer should account for this.

## Pick scorer results (Powered Cube PremierDraft, 2026-11-26)
- Scorer: for each pack card c, gain(c | pool) = w_c + synergy_weight * sum_j pool_j <v_c, v_j>.
  This is the FM's exact change in win logit from adding c to a deck containing the pool.
  It doesn't build a deck yet; the synergy weight stands in for "not every pool card makes
  the deck".
- Evaluation: replays every pick of the held-out drafts (the same draft split as the deck
  model, so no leakage). 17Lands draft data never includes P1P1, so drafts have 44 picks,
  and last picks (one card) are skipped. Test set: 195,119 picks from 4,759 drafts.
- Test results ("wins/agree" is extra match wins per 100% agreement with the scorer's
  top pick, from a per-draft regression with skill, experience and rank controls, using
  robust SE):

  | scorer | top-1 agreement | strong drafters | wins/agree |
  |---|---|---|---|
  | card win rate (17Lands-style) | 0.235 | 0.277 | +2.01 +/- 0.16 |
  | logreg card value | 0.227 | 0.235 | +4.11 +/- 0.42 |
  | FM card value only | 0.225 | 0.234 | +4.56 +/- 0.43 |
  | FM + synergy x1 (default) | 0.238 | 0.248 | +4.73 +/- 0.41 |
  | FM + synergy x2 | 0.251 | 0.259 | +4.77 +/- 0.40 |
  | FM + synergy x8 | 0.282 | 0.288 | +3.16 +/- 0.37 |

- What this means:
  - Following the FM's picks is associated with more than twice the extra wins of
    following raw card win rates (+4.7 vs +2.0, a gap of about 6 SE). This is the main
    evidence that the model ranks picks better.
  - Synergy adds little to the outcome measure over FM value alone (+4.73 vs +4.56, within
    noise). Its effect is clearer in agreement: higher synergy weights pick more like
    humans (top-1 agreement rises from 0.225 to 0.282), but past x2 the win association
    falls. The weight was chosen on validation (x1 had the best val wins/agree, 4.26).
  - Top-1 agreement with humans is low (about 24%) for every scorer. That is expected:
    humans don't maximize this objective, and matching them is not the goal.
  - Observational caveat: drafters who take the scorer's picks may differ in ways the
    skill bucket doesn't capture. Only live or simulated drafts can fully confirm it.
- Sanity checks: the single-pick `rank()` and batch `score_packs()` agree to 1e-8. In a
  sample P2P4 with a UR/Wx pool, the top pick was Ajani, Nacatl Pariah, matching the human.
- Ideas for later: build the likely final deck from the pool (colors plus best 23)
  instead of weighting the whole pool, discount cards outside the pool's colors early,
  and add a pick model trained on the draft data for signals and wheel estimates.

## Arena log findings (from the 2026-05-09 SOS logs)
- `Draft.Notify {"draftId","SelfPack","SelfPick","PackCards":"grpId,..."}` is logged for
  every pack shown, with 1-based numbering. It appears even in the afternoon log, which
  has no pick records. Play boosters have 14 cards, so a full draft has 42 notifies.
- `==> EventPlayerDraftMakePick {"request":"{\"DraftId\",\"GrpIds\":[id],\"Pack\",\"Pick\"}"}`
  records the user's pick. These lines are only in the evening log (Detailed Logs).
- After the last pick, `<== DraftCompleteDraft(...)` is followed by a course JSON with
  InternalEventName and CardPool (42 grpIds). Later course listings repeat it.
- The event hint comes from the "Entering table draft queue: SOS_Premier_Draft" scene change.
- What the logs contain: 5 complete drafts in Player-prev.log with packs only (picks
  can't be recovered). One draft (f2833bbe) spans the restart: its P1 packs are in
  Player-prev.log, and its P2-P3 picks and final pool are in Player.log. All 14 P1 picks
  were inferred uniquely from the pool, using the rule that a picked card is gone when its
  pack wheels 8 picks later. The P2P1 pack notify is in neither file, because Arena
  restarted at that moment.
- On that draft, the SOS scorer's top pick matched the user's on 11/38 picks (29%), close
  to the 24% offline agreement rate.

## Mana-aware scorer (2026-11-27, default)
Goal: fixing, low curve and splashing, done generically (ship target is Powered Cube,
whose card list will change; the live test is TRK).
- Card data: rules text now loaded from Arena's DB (ArenaCard.text). mtgdraft/model/mana.py
  parses pips (hybrid as either color; phyrexian and twobrid ignored) and produced colors
  ("Add {U}", "any color", fetches by basic type, "basic land card" = any; Arcane
  Signet/Command Tower make nothing in limited; a nonland source doesn't count for colors
  it needs itself). Castability = exact hypergeometric P(enough sources of each color
  by turn mv + slack) in a 40-card deck. Card-agnostic, so new cube cards work.
- mtgdraft/model/mana_scorer.py: 70 lanes (10 pairs, 10 triples, 30 pair+splash, 20
  triple+splash). Per lane, a 17-land base: pool nonbasics and castable mana sources making
  lane colors, then basics: splash gets up to splash_basics minus existing fixing, the
  rest split by the pool's pip demand. eff = p*val + (1-p)*dead. Fixers also score the
  lane-strength gain from adding them to the mana base.
- Lane comparison bug found and fixed: comparing lanes by value above each lane's own
  replacement level graded deep pairs on a harder curve (a U/B drafter was put in W/U
  with no white cards). Lanes now use the best-23 deck value over one shared baseline.
- Hand formulas still matched humans' deck colors worse than "pair with the most
  cards" (55% vs 62% on cube), so which lane a pool becomes is now a fitted conditional
  logit over lanes (scripts/fit_lane_model.py, data/models/lane_model.json): 6
  card-agnostic features (castable deck value in SDs, depth of playables, thinnest main
  color, triple, splash, splash cards), fitted on cube + SOS training drafts together.
  Held-out test, main colors matched: cube 69.0% (count baseline 57.8%), SOS 84.9%
  (77.8%). Chosen slack 2, splash basics 2. It under-predicts splashes (17Lands marks a
  single off-color card as a splash).
- Pick evaluation (scripts/eval_picks.py --skip-old --max-drafts 800, held-out test;
  one shared setting t=0.025, q=0.5, chosen on validation by strong drafters' top-1,
  the same values were best for all three sets):

  | set | top-1 deck-aware -> mana-aware | percentile | wins/agree |
  |---|---|---|---|
  | TRK | .329 -> .363 | .680 -> .741 | 2.96 -> 2.66 (+/- 0.8) |
  | SOS | .418 -> .504 | .736 -> .812 | 2.87 -> 1.72 (+/- 0.8) |
  | Cube | .292 -> .358 | .622 -> .706 | 4.05 -> 2.94 (+/- 0.8) |

  Bot-drafted pools (200 test drafts; deck judged by the lane model): nonland cards
  castable in the deck it becomes, humans / deck-aware / mana-aware: TRK 88/85/95%,
  SOS 82/70/89%, cube 87/75/91%. Deck shapes: SOS close to humans (2-color 64% vs 73%);
  TRK goes 3-color more than humans (51% vs 26%; TRK has many fixers: 38 cards make 2+
  colors, Planets and shocks); cube splashes/3-colors less than humans (2-color 86% vs
  56%). Wins/agree is lower on all three test splits but each within about 1.5 SE; on
  validation it was mixed (cube higher).
- Curve: tested deck-structure covariates (spells per mana value, creatures, lands, and
  their squares; mtgdraft/model/structure.py, scripts/eval_structure.py). No held-out
  gain: cube -0.00005 +/- 0.00007, TRK -0.00009 +/- 0.00008, SOS +0.00024 +/- 0.00008
  (worse). Not shipped. The per-card values already carry it: in cube, mean value by
  mana value is +0.28 SD for 1-drops and -0.44 SD for 6+; in TRK it is flat.
- Quick/Traditional drafts use the set's Premier Draft model; lane_model.json is shared.

### Do winners splash differently? (2026-11-27)
- 17Lands deck shapes by match wins: winners build FEWER colors. Cube, 0-2 vs 7 wins:
  2-color 17% vs 26%, 3+splash 17% vs 10%; 2+splash ~45% throughout. Match win rate
  by shape (cube): 2-color 51.6%, 2+splash 48.0%, 3 45.0%, 3+splash 43.3%. SOS: 2-color
  48% -> 58% from 0-2 to 7 wins. (Correlational: weaker pools/players may go wider.)
- Lane model refit on winning-record decks only (fit_lane_model.py --winners --out
  lane_model_winners.json): weights almost identical to the all-decks model, so given a
  pool, winners choose colors the same way; no per-format splash setting needed. Kept
  the all-decks lane_model.json. load_scorer kind "mana@<file>" loads another lane model.
- TRK game data still not published (checked 2026-11-27); TRK draft data already current.

### Bot decks vs human decks (2026-11-27, scripts/compare_decks.py)
Same independent judge, castability-adjusted, neutral controls. Human drafts from the
training split (never seen by the judge); bot pools from 50 simulated pods (400 decks
per scorer). Fair comparison = both built by our deck builder (free / human-style):

| judged win rate | TRK free | TRK human-style | Cube free | Cube human-style |
|---|---|---|---|---|
| bot, new scorer | 42.7% | 40.8% | 56.6% | 54.7% |
| bot, old scorer | 39.6% | 35.7% | 55.8% | 53.2% |
| humans, all | 43.6% | 41.5% | 58.9% | 57.0% |
| humans, winning record | 43.9% | 41.7% | 60.0% | 58.2% |
| humans, 7 wins | 45.2% | 43.2% | 60.6% | 59.0% |

New-scorer bot decks are about 1 point below the average human deck in TRK and about 2
points below in cube; 1-2 (TRK) and 3-4 (cube) points below winning-record decks. The
judge separates human decks by record only weakly (deck quality is part of winning;
play skill is the rest). Bot pools come from all-bot pods, where all 8 seats chase the
same cards; humans' varied preferences leave more good cards per seat, so this likely
understates the bot. "As played" human decks rate lower than our builds of the same
pools (the builder picks the judge's best build; TRK decks rebuilt from draft data are
also incomplete), so compare the builder rows.

### A/B test in simulated pods (2026-11-27, scripts/ab_test.py)
- 100 pods per set, 8 bot seats alternating mana-aware (A) / deck-aware (B), each pod
  replayed with seats swapped; 24 real opening packs per pod from held-out 17Lands drafts.
- Judge: an FM trained only on val+test drafts (data/models/judge.<set>.<event>.npz),
  which the scorers' model never saw. Decks: one neutral builder for both; each spell's
  uncastable share (1 - p) is valued as a dead card (5th-percentile value), because the
  judge alone happily plays uncastable cards (its training decks never have any). Two
  builds: "free" (colors the judge likes best) and "human-style" (colors from the
  lane-choice model, then the best 23).
- Judged win rate, A - B (SE over pods):

  | set | free build | human-style build | scorers' own model (free) |
  |---|---|---|---|
  | TRK | +2.77 +/- 0.47 pts | +4.53 +/- 0.47 | +3.29 +/- 0.34 |
  | Cube | +1.59 +/- 0.43 | +2.46 +/- 0.44 | +0.27 +/- 0.29 |

  Deck spells' mean castability A vs B: TRK 0.96 vs 0.86, cube 0.98 vs 0.89. Lands picked
  per pool: TRK 5.7 vs 3.0, cube 10.6 vs 6.5 (the new scorer takes fixing much earlier).
- Caveats: win rates are the judge's predictions, not games; all seats are bots (no human
  signals); the castability math and the lane-choice model are shared by scorer A and the
  builder (the free build and the judge are independent of A). Bot-only pods also make
  free-built decks wide (3+splash 47-61% in TRK), more than humans build.

## Deck-aware scorer (2026-11-26, previous default)
- Problem found: the plain scorer (card value + synergy with the whole pool) was
  color-blind. Bot-drafted SOS pools had only 50% of nonland cards castable in their best
  two colors (humans: 63% on the user's draft), because off-color pair penalties are about
  20x smaller than card values, and every pool card counted as if it would be played.
- Fix: `mtgdraft/model/deck_scorer.py`. For each of the 10 color pairs, project the
  pool's best 23 castable spells (plus that pair's basics, whose learned pairings give a
  color signal), weight pairs by softmax(deck strength / temperature), and score a
  candidate by its softplus gain over the cut line (replacement level while slots are
  open, else the weakest deck card), weighted across pairs. Off-color cards and cards that
  wouldn't make the deck score near zero; mono-color cards fit 4 of 10 pairs and gold
  cards 1 early on. Lands must beat replacement level too.
- Tuned on validation drafts: temperature 0.05, replacement quantile 0.7.
- Card values are shrunk by evidence when scoring: w * n/(n+50), with n the number of
  training decks that played the card (saved in the model as card_decks). This stops
  barely-played cards (e.g. TRK's Bruvac, 2 decks) from topping the list.
- Held-out test results (eval_picks.py; color fit = share of the bot's own pool castable
  in its best pair, bot drafting along the packs humans saw, 300 drafts):

  | set | scorer | top-1 agreement | color fit (humans) | wins/agree |
  |---|---|---|---|---|
  | TRK | plain | 0.268 | 59% (79%) | +3.82 +/- 0.51 |
  | TRK | deck-aware | 0.330 | 73% | +3.58 +/- 0.45 |
  | SOS | plain | 0.327 | 53% (76%) | +2.57 +/- 0.23 |
  | SOS | deck-aware | 0.416 | 63% | +2.84 +/- 0.20 |
  | Cube | plain | 0.238 | 57% (77%) | +4.76 +/- 0.41 |
  | Cube | deck-aware | 0.293 | 75% | +4.38 +/- 0.34 |

  The deck-aware scorer agrees with humans much more often and commits to colors much
  better. Its association with winning is about the same as the plain scorer's (within
  noise). The color-fit check is approximate: later packs follow the human's picks.

## Star Trek (TRK) support (2026-11-26)
- 17Lands publishes TRK draft data only (Premier, Trad, PickTwo; Nov 10-22, 23,785 Premier
  drafts). No game data yet.
- `games.load_draft_decks()` rebuilds decks from draft data: a pick is a deck card if its
  pick_maindeck_rate >= 0.5 (fractional rates inflated rare cards' weights), and each
  draft's match record becomes one row per match. `load_deck_data()` uses game data when
  it exists and falls back to this. No basics in these decks.
- TRK model: 123,725 match rows / 22,701 drafts / 341 cards. Test log-loss: logreg 0.67458,
  FM (k=16, l2_v=0.01) 0.67438 (-0.00020 +/- 0.00009). Small data, so synergy needs strong
  regularization (default l2_v grid widened to 1e-3..3e-2).
- TRK draft data includes P1P1 (42 picks per draft), unlike the older cube data.
- Arena's card DB (updated 2026-11-26) has all 447 TRK printings; all 341 model names match.
- Scryfall has no Arena IDs for TRK yet; card art is looked up by name + set (trk).

## Live assistant and overlay (2026-11-26)
- `scripts/draft_assistant.py --top N`: follows the Arena log, ranks each pack with the
  deck-aware scorer, shows an always-on-top advisor panel, and blacks out all but the top N
  cards on screen. Archives earlier logs at start. Auto-downloads 17Lands data and trains a
  model if the set has none.
- Card finding (`mtgdraft/overlay/locate.py`): the log says which cards are in the pack;
  each card's Scryfall art_crop is matched to a screenshot with SIFT + RANSAC homography,
  then expanded to the full card with frame proportions (normal and Saga frames). No fixed
  positions. Synthetic test (`scripts/test_locator.py`): 158/160 cards boxed (IoU >= 0.7),
  0 wrong boxes, at 720p to 4K and 115-360 px wide cards, ~0.65 s per screenshot.
- Overlay window (`mtgdraft/overlay/window.py`): borderless, click-through
  (WS_EX_TRANSPARENT), no-activate, topmost (re-asserted on every draw; note HWND_TOPMOST
  must be passed as a pointer-sized handle on 64-bit), excluded from screen capture
  (WDA_EXCLUDEFROMCAPTURE) so it never hides cards from the locator. Boxes are sticky per
  pick. DPI-aware, so screenshots and windows share physical pixels.
- End-to-end test passed: a held-out 17Lands TRK draft fed as a fake live log, a fake Arena
  window showing the packs; every pick located 8/8 to 13/14 cards and the overlay blacked
  out all but the top 3 with rank badges.
- Arena on this PC is set to borderless fullscreen (Unity "Fullscreen mode" = 1) at
  1920x1080, which overlays can draw over. Exclusive fullscreen (0) would hide them.
- Not yet verified against the real Arena draft screen (needs a live draft): Arena's card
  rendering or special printings may differ from Scryfall art.

## Where the Powered Cube logs went (checked 2026-11-26)
- No Powered Cube logs exist on this PC. Arena's Player-prev.log was the 2026-05-09 SOS
  session until Arena was launched on 2026-11-26, so Arena was not run on this Windows
  account in between. Untapped.gg Companion's saved drafts (drafts.json) are the same 6
  SOS drafts. The PC's other Windows account has no Arena data.
- The cube drafts were most likely played on another device; the user's 17Lands or
  Untapped.gg account may have them online.
- `scripts/archive_logs.py` now keeps every session (gzip, logs/archive/); the live
  assistant archives at start. Arena was relaunched at ~11:19 on 2026-11-26, which
  overwrote the May 9 log in Arena's folder; the project's copies are in the archive.

## Known gotchas
- P1P1: RESOLVED for Arena logs. Draft.Notify includes P1P1 with all 14 cards (checked on
  all 6 drafts). It is the 17Lands draft dataset that never includes P1P1. Recheck on the
  first live draft.
- A pack notify can be lost if Arena restarts mid-draft. The live overlay then has no
  pack until the next pick. Fallback: the screenshot/vision path.
- The batch evaluator (`score_packs`) still scores basic lands. It is harmless for the
  cube (no basics in packs), but for regular sets it should match `rank()`.

## 17Lands data access (verified 2026-11-26)
- The dataset list is JSON at https://www.17lands.com/api/public_datasets
  (the /public_datasets page is rendered by JavaScript).
- Files are served from data.17landscapes.com/analysis_data/{draft_data,game_data,replay_data}/
  {type}_data_public.{SET}.{EventType}.csv.gz
- The data is released under CC BY 4.0, so credit 17Lands.
- Not every set has every file type. Some have only draft data, some only game data.
