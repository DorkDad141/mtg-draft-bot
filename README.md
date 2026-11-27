# mtg-draft-bot

A draft assistant for MTG Arena Premier Draft. It reads the current pack from Arena's
Player.log, ranks each card by how much it raises the expected win rate of the deck being
built, and draws an overlay that hides all but the top N cards.

See [docs/PROJECT_NOTES.md](docs/PROJECT_NOTES.md) for design decisions and results,
and [docs/TRK_TEST.md](docs/TRK_TEST.md) for a step-by-step live test.

## Setup

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

## Quick start: live draft assistant

```
.venv\Scripts\python scripts/draft_assistant.py --top 3
```

Start it, then draft in Arena (windowed or borderless fullscreen; enable *Detailed
Logs (Plugin Support)*). For every pack it shows an always-on-top advisor panel
(projected deck colors and the pack ranked) and blacks out every card on screen
except the top N, which get rank badges. Cards are found on screen by matching
their art (Scryfall images, cached in `data/card_images/`), so it works at any
resolution or scaling; the overlay is click-through and invisible to screenshots.
If the draft's set has no model yet, it downloads 17Lands data and trains one.
From each pick until the next pack is ranked, the whole pack area stays black, so
you never see the next pack unranked. Premier, Traditional and Quick Draft logs are all
recognised automatically; Quick/Traditional drafts use the set's Premier Draft model. Live sessions are recorded to `captures/<date_time>/`
(screenshots before and after the blackout, rankings, picks and the Arena log) for
future training; `--no-record` turns that off. When a recorded draft ends, the
assistant proposes and rates your deck (shown in the advisor) and builds a pick-by-pick
viewer with a final deck slide (`scripts/propose_deck.py`, `scripts/draft_viewer.py`).

Rehearse without Arena or a paid draft, using a real 17Lands draft as a fake live
log and a fake Arena window:

```
.venv\Scripts\python scripts/draft_assistant.py --top 3 --sim-17lands TRK --fake-screen
```

Options: `--no-overlay` (advisor only), `--console` (text only), `--opacity 0.85`,
`--no-shield` (only black out the located non-top cards).

## Getting the data

Requires Python 3.10+ (the downloader needs no extra packages).

```
python scripts/download_17lands.py --list
python scripts/download_17lands.py --set "Powered Cube" --event PremierDraft
python scripts/download_17lands.py --set FDN --event PremierDraft --types draft game
```

Files go to `data/raw/` (git-ignored). Rerunning only downloads files that 17Lands has
updated; `--force` downloads again anyway.

## Training the deck model

```
.venv\Scripts\python scripts/train_deck_model.py --set Cube_-_Powered --event PremierDraft
```

Trains a factorization machine that predicts a deck's win rate from its cards
(card values plus pairwise synergy), compares it against simpler baselines on
held-out drafts, and saves the best model to `data/models/`. Uses 17Lands game
data when it exists; for a brand-new set with only draft data, it rebuilds decks
from each drafter's main-deck picks and uses their match record.

## Evaluating picks

```
.venv\Scripts\python scripts/eval_picks.py --set Cube_-_Powered --event PremierDraft
```

Replays held-out 17Lands drafts and compares how each scorer ranks the packs the
drafter saw: agreement with the human pick, whether following the scorer is
associated with more match wins, and how color-disciplined the bot's own pools are.

`scripts/test_locator.py --set TRK` tests the on-screen card finder on synthetic
screens at several resolutions.

## Replaying Arena drafts

```
.venv\Scripts\python scripts/show_draft.py path\to\Player-prev.log path\to\Player.log
.venv\Scripts\python scripts/show_draft.py --live
```

Reads Arena's Player.log (enable *Detailed Logs (Plugin Support)* in Arena's
settings so picks are logged), maps card IDs to names with Arena's local card
database, and shows the scorer's top cards for every pack. A model for the
draft's set must have been trained first (`train_deck_model.py --set SOS`).
`--live` follows the running Arena log.

## Archiving Arena logs

Arena keeps only the current and previous session's log (`Player.log`,
`Player-prev.log`) and overwrites the older one each time it starts, so drafts
older than two launches are lost. Archive them:

```
.venv\Scripts\python scripts/archive_logs.py            # copy both logs now
.venv\Scripts\python scripts/archive_logs.py --list     # what's archived, with draft events
.venv\Scripts\python scripts/archive_logs.py --watch 10 # keep archiving every 10 minutes
```

Sessions are saved as `logs/archive/Player_<start time>.log.gz` (git-ignored; logs
contain your account name). A running session's copy is refreshed as it grows.
`show_draft.py --live` also archives on start. The replay tools read `.gz` files
directly, e.g. `show_draft.py logs\archive\*.gz`.

To archive automatically every 10 minutes, register a Windows scheduled task
(run this yourself in PowerShell from the repo folder; remove it later with
`Unregister-ScheduledTask "MTG draft log archiver"`):

```
$repo = (Get-Location).Path
$a = New-ScheduledTaskAction -Execute "$repo\.venv\Scripts\pythonw.exe" -Argument "`"$repo\scripts\archive_logs.py`""
$t = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Minutes 10)
Register-ScheduledTask -TaskName "MTG draft log archiver" -Action $a -Trigger $t
```

## Scorers

- **mana** (default): mana-aware. Considers 70 possible decks (each color pair, each
  three-color deck, and each of those with a splash). For each it builds a 17-land mana
  base from the pool (duals, fetches, Signets, Moxen and dorks are recognised from
  their rules text) and values every card by its chance of being castable, using exact
  hypergeometric odds. So a single-pip bomb can be splashed, double-pip cards can't,
  and fixing is valued by how much castability it adds to the pool. Which deck a pool
  becomes is predicted by a small lane-choice model fitted on what humans actually
  built (cube and SOS game data; 6 shared weights, applied to every set). Build it
  with `python scripts/fit_lane_model.py` (data/models/lane_model.json).
- **deck**: the older deck-aware scorer. Projects the pool onto the ten two-color decks it
  could become, weights them by strength, and scores each card by how much it
  improves those likely decks beyond a replacement-level card (two-color decks only,
  no splashes). Needs Arena's card database for card colors.
- **plain**: card value plus synergy with every card in the pool, regardless of color.

`show_draft.py` and `simulate_live_log.py` take `--scorer mana|deck|plain`. Calibration
and checks: `scripts/eval_mana.py`, `scripts/fit_lane_model.py`, `scripts/eval_structure.py`,
and `scripts/eval_picks.py` (which compares the deck-aware and mana-aware scorers).
`scripts/ab_test.py --set TRK --pods 100` runs an A/B test of two scorers in simulated
8-player pods, with decks judged by a model trained only on held-out drafts.

## Simulating a live draft

```
.venv\Scripts\python scripts/simulate_live_log.py logs\Player-prev.log logs\Player.log --list
.venv\Scripts\python scripts/simulate_live_log.py logs\Player-prev.log logs\Player.log --draft f283
.venv\Scripts\python scripts/simulate_live_log.py logs\Player-prev.log logs\Player.log --draft 2165 --bot-picks
```

Replays a recorded draft into `data/sim/Player.log` in Arena's log format, one
pack and pick at a time, while the live reader follows it. Each pack is shown
with every card ranked by the current model, then the pick actually made and
where the bot ranked it. `--bot-picks` lets the bot make picks the log doesn't
have; `--delay` sets the pace; `--top N` shortens the lists. `--report` also writes
an HTML page (`data/reports/draft_<id>.html`) with every step: the pool so far,
the pack, and the bot's full ranking split into card value and pool synergy. With `--no-watch` it
only writes the file, so `show_draft.py --live data\sim\Player.log` in a second
terminal sees it exactly as it would see Arena.

## Credits

Draft and game data: [17Lands](https://www.17lands.com/public_datasets) public datasets,
licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Card images for the on-screen card finder: [Scryfall](https://scryfall.com) (cached
locally, not redistributed).
