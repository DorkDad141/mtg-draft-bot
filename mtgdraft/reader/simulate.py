"""Build simulated Arena Player.log lines, for testing the live path without a real draft.

Two sources:
  * a recorded Arena DraftState (see scripts/simulate_live_log.py)
  * a 17Lands draft (draft data rows), so any set with 17Lands data can be "drafted"
    offline, including brand-new sets you have no logs for yet.
"""

from __future__ import annotations

import json
import uuid

PREFIX = "[UnityCrossThreadLogger]"


def queue_line(event_name: str) -> str:
    j = {"fromSceneName": "EventLanding", "toSceneName": "TableDraftQueue", "initiator": "System",
         "context": f"Entering table draft queue: {event_name}"}
    return f"{PREFIX}Client.SceneChange {json.dumps(j)}"


def notify_line(draft_id: str, slot, cards) -> str:
    j = {"draftId": draft_id, "SelfPick": slot[1], "SelfPack": slot[0],
         "PackCards": ",".join(str(c) for c in cards)}
    return f"{PREFIX}Draft.Notify {json.dumps(j, separators=(',', ':'))}"


def pick_line(draft_id: str, slot, card: int) -> str:
    req = {"DraftId": draft_id, "GrpIds": [card], "Pack": slot[0], "Pick": slot[1]}
    j = {"id": str(uuid.uuid4()), "request": json.dumps(req, separators=(',', ':'))}
    return f"{PREFIX}==> EventPlayerDraftMakePick {json.dumps(j, separators=(',', ':'))}"


def bot_status_lines(event_name: str, slot, cards, picked, request: bool = False,
                     completed: bool = False) -> list[str]:
    """Quick Draft: the pack shown (as a DraftStatus/DraftPick response). slot is 1-based."""
    rid = str(uuid.uuid4())
    payload = {"Result": "Success", "EventName": event_name,
               "DraftStatus": "Completed" if completed else "PickNext",
               "PackNumber": slot[0] - 1, "PickNumber": slot[1] - 1, "NumCardsToPick": 0 if completed else 1,
               "DraftPack": [] if completed else [str(c) for c in cards], "PackStyles": [],
               "PickedCards": [str(c) for c in picked], "PickedStyles": []}
    resp = {"CurrentModule": "BotDraft", "Payload": json.dumps(payload, separators=(',', ':'))}
    out = []
    if request:
        req = {"id": rid, "request": json.dumps({"EventName": event_name}, separators=(',', ':'))}
        out.append(f"{PREFIX}==> BotDraft_DraftStatus {json.dumps(req, separators=(',', ':'))}")
    name = "BotDraft_DraftStatus" if request else "BotDraft_DraftPick"
    out += [f"<== {name}({rid})", json.dumps(resp, separators=(',', ':'))]
    return out


def bot_pick_line(event_name: str, slot, card: int) -> str:
    info = {"EventName": event_name, "CardIds": [str(card)], "PackNumber": slot[0] - 1, "PickNumber": slot[1] - 1}
    req = {"EventName": event_name, "PickInfo": info}
    j = {"id": str(uuid.uuid4()), "request": json.dumps(req, separators=(',', ':'))}
    return f"{PREFIX}==> BotDraft_DraftPick {json.dumps(j, separators=(',', ':'))}"


def pack_in_line(line: str) -> list[int] | None:
    """The pack's grpIds if this simulated line shows a pack (either draft kind), else None."""
    if "Draft.Notify" in line:
        j = json.loads(line[line.index("{"):])
        return [int(x) for x in j["PackCards"].split(",") if x]
    if line.startswith("{") and '"BotDraft"' in line:
        p = json.loads(json.loads(line)["Payload"])
        return [int(x) for x in p.get("DraftPack") or []] or None
    return None


def pool_lines(event_name: str, pool: list[int]) -> list[str]:
    rid = str(uuid.uuid4())
    course = {"CourseId": str(uuid.uuid4()), "InternalEventName": event_name,
              "CurrentModule": "DeckSelect", "ModulePayload": "", "CardPool": pool}
    return [f"<== DraftCompleteDraft({rid})", json.dumps(course, separators=(',', ':'))]


def steps_from_17lands(rows, grp_of: dict[str, int], event_name: str, pick_delay: float = 1.0,
                       quick: bool = False):
    """Simulated log steps [(delay_before, line)] from one 17Lands draft.

    rows: dicts/Series with pack_number, pick_number (0-based), pick, and pack_card_<name>
    counts (as in 17Lands draft data). grp_of maps card names to Arena grpIds.
    quick: write Quick Draft (BotDraft) lines instead of Premier Draft ones.
    """
    if quick:
        return _bot_steps(rows, grp_of, event_name, pick_delay)
    draft_id = str(uuid.uuid4())
    steps = [(0.0, queue_line(event_name))]
    pool = []
    for r in sorted(rows, key=lambda r: (r["pack_number"], r["pick_number"])):
        slot = (int(r["pack_number"]) + 1, int(r["pick_number"]) + 1)
        cards = []
        for k, v in r.items():
            if k.startswith("pack_card_") and v and k[10:] in grp_of:
                cards += [grp_of[k[10:]]] * int(v)
        if not cards:
            continue
        steps.append((0.0, notify_line(draft_id, slot, cards)))
        pick = grp_of.get(r["pick"])
        if pick is not None:
            pool.append(pick)
            steps.append((pick_delay, pick_line(draft_id, slot, pick)))
    steps += [(0.0, line) for line in pool_lines(event_name, pool)]
    return draft_id, steps


def _bot_steps(rows, grp_of, event_name, pick_delay):
    steps, picked, first = [], [], True
    last = (1, 1)
    for r in sorted(rows, key=lambda r: (r["pack_number"], r["pick_number"])):
        slot = (int(r["pack_number"]) + 1, int(r["pick_number"]) + 1)
        cards = []
        for k, v in r.items():
            if k.startswith("pack_card_") and v and k[10:] in grp_of:
                cards += [grp_of[k[10:]]] * int(v)
        if not cards:
            continue
        steps += [(0.0, ln) for ln in bot_status_lines(event_name, slot, cards, picked, request=first)]
        first = False
        pick = grp_of.get(r["pick"])
        if pick is not None:
            steps.append((pick_delay, bot_pick_line(event_name, slot, pick)))
            picked = picked + [pick]
        last = slot
    steps += [(0.0, ln) for ln in bot_status_lines(event_name, last, [], picked, completed=True)]
    return f"{event_name}#1", steps
