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

## Credits

Draft and game data: [17Lands](https://www.17lands.com/public_datasets) public datasets,
licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
