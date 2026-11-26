# mtg-draft-bot

A draft assistant for MTG Arena Premier Draft. It reads the current pack from Arena's
Player.log, ranks each card by how much it raises the expected win rate of the deck being
built, and draws an overlay that hides all but the top N cards.

Work in progress. See [docs/PROJECT_NOTES.md](docs/PROJECT_NOTES.md) for the plan.

## Setup

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

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
held-out drafts, and saves the best model to `data/models/`.

## Evaluating picks

```
.venv\Scripts\python scripts/eval_picks.py --set Cube_-_Powered --event PremierDraft
```

Replays held-out 17Lands drafts and compares how each scorer ranks the packs the
drafter saw: agreement with the human pick, and whether following the scorer is
associated with more match wins.

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
have; `--delay` sets the pace; `--top N` shortens the lists. With `--no-watch` it
only writes the file, so `show_draft.py --live data\sim\Player.log` in a second
terminal sees it exactly as it would see Arena.

## Credits

Draft and game data: [17Lands](https://www.17lands.com/public_datasets) public datasets,
licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
