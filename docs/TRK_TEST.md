# Testing the whole assistant on a Star Trek (TRK) draft

Everything below runs from the repo folder (`%USERPROFILE%\mtg-draft-bot\repo`) in
PowerShell. Steps 1–2 cost nothing; step 3 is the real (paid) draft.

## What is ready
- TRK deck model, trained from 17Lands TRK Premier Draft data (22,701 drafts,
  Nov 10–22). 17Lands has no TRK *game* data yet, so decks are rebuilt from each
  drafter's main-deck picks and match record. Retrain any time with
  `.venv\Scripts\python scripts/download_17lands.py --set TRK --event PremierDraft --types draft game`
  then `.venv\Scripts\python scripts/train_deck_model.py --set TRK`.
- Card art for all 311 TRK cards is cached for the on-screen card finder.
- Arena is set to borderless fullscreen 1920x1080 (checked in the registry), which
  the overlay can draw over. Detailed Logs is enabled.

## 1. Rehearsal with a fake Arena screen (2 minutes, no Arena needed)
```
.venv\Scripts\python scripts/draft_assistant.py --top 3 --sim-17lands TRK --fake-screen
```
A real held-out 17Lands TRK draft plays as a fake live log. A "Fake Arena" window
shows each pack; the advisor panel (top right) ranks it; the 11 cards outside the
top 3 get blacked out, and the top 3 get #1/#2/#3 badges. Close the advisor window
to stop. `--sim-index 5` picks a different draft; `--sim-delay 5` slows it down.

## 2. Check the overlay can draw over Arena (optional, no draft needed)
Start Arena, leave it on any screen, then run the rehearsal again. The advisor and
badges should appear over Arena. If nothing shows over Arena, switch Arena to
Windowed or borderless (Options > Graphics), since exclusive fullscreen hides overlays.

## 3. The live TRK draft
1. Start Arena (Detailed Logs on: Options > Account > Detailed Logs (Plugin Support)).
2. In PowerShell:
   ```
   .venv\Scripts\python scripts/draft_assistant.py --top 3
   ```
   It archives earlier Arena logs, then follows the live log.
3. Join TRK Premier Draft. On every pack the advisor shows the projected deck
   (most likely color pairs) and the ranked pack; on screen, all but the top 3
   cards are blacked out. Clicks go straight through the overlay to Arena, so you
   can still pick any card.
4. After the draft, close the advisor. Then make the pick-by-pick report:
   ```
   .venv\Scripts\python scripts/simulate_live_log.py logs\archive\*.gz --report --delay 0
   ```
   (or pass Arena's `Player.log` directly), and open `data\reports\draft_<id>.html`.

Options: `--top N` (how many stay visible), `--opacity 0.85` (dim instead of solid
black), `--no-overlay` (advisor panel only), `--console` (text only).

## If something goes wrong
- **Advisor says "No model for ..."**: the event's set code wasn't TRK or the model
  file is missing; it will try to download and train automatically (1–2 minutes).
- **Advisor works but no black boxes**: check the advisor's status line. "Found 0/14
  cards" means the card finder can't see the pack. Arena must be visible (not
  minimized). Scryfall art can differ from Arena's for special printings; those
  cards just stay uncovered.
- **Nothing at all after the first pack**: check Detailed Logs is on; the log is at
  `%USERPROFILE%\AppData\LocalLow\Wizards Of The Coast\MTGA\Player.log`.
- Send the console output and `data\reports\...html`; every session's log is kept in
  `logs\archive\`, so nothing is lost.
