# mtg-draft-bot

A draft assistant for MTG Arena Premier Draft, to be extended to MTGO later. It reads the
current pack and the user's picks from Arena's Player.log. It ranks the pack by how much
each card raises the expected win rate of the deck being built, and shows an overlay that
blacks out everything except the top N cards (N is chosen at launch).

Read `docs/PROJECT_NOTES.md` first. It holds the decisions made so far, the open questions
to revisit later, and the data source details. Keep it up to date when decisions change.

## Working rules
- Python. Training runs on a local CPU (the user's Windows PC); no GPU.
- Minimize paid Arena drafts. Train and evaluate offline on 17Lands data and the user's own
  logs. Live drafts are only for final overlay checks.
- Keep the four modules separate so MTGO can be added by writing a new reader and overlay:
  draft-state reader / card database / ranking model / overlay.
- Do not commit Player.log files. They contain the user's account name and ID. Keep them
  in a git-ignored folder such as `logs/`.
- Downloaded data goes in `data/`, which is git-ignored. The download script must be
  rerunnable so each set's data can be refreshed.
- Credit 17Lands (CC BY 4.0) in the README.
