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

## Credits

Draft and game data: [17Lands](https://www.17lands.com/public_datasets) public datasets,
licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
