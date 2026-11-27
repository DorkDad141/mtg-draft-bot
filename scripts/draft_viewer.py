"""Make an HTML viewer of a recorded live draft: one screenshot per pick, Prev / Next.

For each pick it shows the screen (toggle: blacked out as you saw it, or the full
screen), the pack's ranking with your pick marked, and your picks so far grouped by
color. Reads the session folders in captures/ (a restarted assistant makes several
folders for one draft; they are merged). The page is written next to them and loads
the screenshots by relative path, so it stays a local file (screenshots show the
account name).

    python scripts/draft_viewer.py                    # newest draft in captures/
    python scripts/draft_viewer.py --draft QuickDraft_TRK
    python scripts/draft_viewer.py --site ../draft-viewers   # also a shareable copy
Keys: Left/Right = previous/next pick, T = blacked out / full screen.

--site DIR writes a self-contained copy for a static web host (e.g. GitHub Pages):
DIR/<draft>/index.html plus the screenshots (1280 px wide JPEGs), and a DIR/index.html
listing every draft in DIR. Only the page and screenshots are copied, never the
Arena log.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mtgdraft.carddb import ArenaCardDB  # noqa: E402

CAPTURES = REPO_ROOT / "captures"
WIN_PCT = 100 * 0.2475          # score (logit) -> approximate win-rate points, as in the advisor


def load_sessions(draft_filter: str | None):
    """All events by draft id across session folders, and the image for each (draft, slot)."""
    drafts: dict[str, dict] = {}
    for folder in sorted(p for p in CAPTURES.iterdir() if p.is_dir()):
        ev_path = folder / "events.jsonl"
        if not ev_path.exists():
            continue
        for line in ev_path.read_text(encoding="utf-8").splitlines():
            e = json.loads(line)
            did = e.get("draft_id")
            if not did or (draft_filter and draft_filter not in did):
                continue
            d = drafts.setdefault(did, {"packs": {}, "picks": {}, "frames": {}, "complete": None, "last_t": 0})
            d["last_t"] = max(d["last_t"], e["t"])
            slot = tuple(e["slot"]) if "slot" in e else None
            if e["kind"] == "pack":
                d["packs"][slot] = e
            elif e["kind"] == "pick":
                d["picks"][slot] = e
            elif e["kind"] == "frame":
                prev = d["frames"].get(slot)
                if prev is None or len(e["found"]) >= len(prev[1]["found"]):
                    d["frames"][slot] = (folder.name, e)
            elif e["kind"] == "complete":
                d["complete"] = e
    return drafts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--draft", help="Part of the draft id (default: the most recent draft)")
    ap.add_argument("--site", type=Path, help="Also write a shareable copy into this folder")
    args = ap.parse_args()
    drafts = load_sessions(args.draft)
    if not drafts:
        sys.exit("No recorded drafts found in captures/.")
    did, d = max(drafts.items(), key=lambda kv: kv[1]["last_t"])
    info = ArenaCardDB().by_name()
    db_names = {}
    slots = sorted(set(d["packs"]) | set(d["picks"]))
    steps, pool = [], []
    for slot in slots:
        pk, pick = d["packs"].get(slot), d["picks"].get(slot)
        frame = d["frames"].get(slot)
        pick_name = pick["name"] if pick else None
        ranking = []
        if pk:
            for i, r in enumerate(pk["ranking"], 1):
                ranking.append({"rank": i, "name": r["name"],
                                "score": None if r["score"] is None else round(WIN_PCT * r["score"], 2),
                                "lane": r.get("lane", ""), "top": r["name"] in pk["top"]})
        img = None
        if frame:
            folder, e = frame
            stem = e["image"].removesuffix("_before.jpg")
            img = {"before": f"{folder}/{stem}_before.jpg", "after": f"{folder}/{stem}_after.jpg"}
        steps.append({"slot": f"P{slot[0]}P{slot[1]}", "pack_size": len(pk["pack"]) if pk else None,
                      "lanes": [[p, round(w * 100)] for p, w in (pk["lanes"] if pk else [])],
                      "ranking": ranking, "pick": pick_name, "bot_rank": pick.get("bot_rank") if pick else None,
                      "pool": list(pool), "img": img})
        if pick_name:
            pool.append(pick_name)
    for nm in set(pool) | {r["name"] for s in steps for r in s["ranking"]}:
        c = info.get(nm)
        if c is not None:
            db_names[nm] = {"colors": c.colors, "mv": c.mana_value, "land": "Land" in c.types,
                            "cost": c.mana_cost}
    event = next(iter(d["packs"].values()))["event_name"] if d["packs"] else did
    final = d["complete"]
    safe = "".join(ch if ch.isalnum() else "_" for ch in did).strip("_")
    deck_path = CAPTURES / f"deck_{safe}.json"
    deck = json.loads(deck_path.read_text()) if deck_path.exists() else None
    data = {"draft": did, "event": event, "steps": steps, "cards": db_names,
            "final_lanes": final["lanes"] if final else None, "deck": deck}
    out = CAPTURES / f"viewer_{safe}.html"
    out.write_text(PAGE.replace("__TITLE__", html.escape(f"{event} - pick by pick"))
                   .replace("__DATA__", json.dumps(data).replace("</", "<\\/")), encoding="utf-8")
    n_img = sum(1 for s in steps if s["img"])
    print(f"Wrote {out} ({len(steps)} picks, {n_img} with screenshots"
          + (", plus the proposed deck)" if deck else "; run scripts/propose_deck.py to add a deck slide)"))
    if args.site:
        write_site(args.site, safe, event, data)
    return 0


def write_site(site: Path, safe: str, event: str, data: dict) -> None:
    """Shareable copy: <site>/<draft>/index.html + img/*.jpg (1280 px), and <site>/index.html."""
    import copy

    import cv2
    d = copy.deepcopy(data)
    dest = site / safe
    (dest / "img").mkdir(parents=True, exist_ok=True)
    size = 0
    for s in d["steps"]:
        if not s["img"]:
            continue
        for kind in ("before", "after"):
            src = CAPTURES / s["img"][kind]
            name = f"{s['slot']}_{'full' if kind == 'before' else 'blacked'}.jpg"
            im = cv2.imread(str(src))
            if im is None:
                continue
            h, w = im.shape[:2]
            if w > 1280:
                im = cv2.resize(im, (1280, round(h * 1280 / w)), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(dest / "img" / name), im, [cv2.IMWRITE_JPEG_QUALITY, 85])
            size += (dest / "img" / name).stat().st_size
            s["img"][kind] = f"img/{name}"
    (dest / "index.html").write_text(PAGE.replace("__TITLE__", html.escape(f"{event} - pick by pick"))
                                     .replace("__DATA__", json.dumps(d).replace("</", "<\\/")), encoding="utf-8")
    links = sorted(p.name for p in site.iterdir() if (p / "index.html").exists())
    items = "\n".join(f'<li><a href="{html.escape(n)}/">{html.escape(n.replace("_", " "))}</a></li>' for n in links)
    (site / "index.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Draft viewers</title>"
        "<style>body{font:15px/1.5 'Segoe UI',system-ui,sans-serif;background:#15161a;color:#e8e8ec;padding:24px}"
        "a{color:#7fb2ff}</style><h1>Recorded drafts</h1><ul>" + items + "</ul>", encoding="utf-8")
    (site / ".nojekyll").write_text("")
    print(f"Wrote shareable copy to {dest} ({size / 1e6:.1f} MB of screenshots) and {site / 'index.html'}")


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>__TITLE__</title>
<style>
:root { --bg:#15161a; --panel:#1f2026; --line:#30323a; --text:#e8e8ec; --dim:#9a9ca6; --gold:#ffd75e; --accent:#7fb2ff; }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--text); font:14px/1.4 "Segoe UI", system-ui, sans-serif; }
header { display:flex; align-items:center; gap:14px; padding:10px 16px; border-bottom:1px solid var(--line); flex-wrap:wrap; }
header h1 { font-size:16px; margin:0 8px 0 0; font-weight:600; }
button { background:#2b2d35; color:var(--text); border:1px solid #444752; border-radius:6px; padding:6px 14px; font:inherit; cursor:pointer; }
button:hover { background:#353845; } button:disabled { opacity:.4; cursor:default; }
button.on { background:#3a4a66; border-color:var(--accent); }
#where { font-weight:600; min-width:70px; text-align:center; }
#bar { flex:1; min-width:160px; height:6px; background:#2b2d35; border-radius:3px; overflow:hidden; }
#bar div { height:100%; background:var(--accent); }
main { display:grid; grid-template-columns: minmax(0, 1fr) 340px; gap:14px; padding:14px 16px; }
.shot { background:#000; border:1px solid var(--line); border-radius:8px; overflow:hidden; position:relative; }
.shot img { display:block; width:100%; height:auto; }
.shot .none { padding:80px 20px; text-align:center; color:var(--dim); }
.shot .tag { position:absolute; top:8px; left:8px; background:rgba(0,0,0,.7); padding:3px 8px; border-radius:4px; font-size:12px; }
aside { display:flex; flex-direction:column; gap:12px; }
.card { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:10px 12px; }
.card h2 { font-size:13px; margin:0 0 6px; color:var(--dim); font-weight:600; text-transform:uppercase; letter-spacing:.04em; }
table { width:100%; border-collapse:collapse; font-size:13px; }
td { padding:2px 4px; vertical-align:top; } td.num { text-align:right; color:var(--dim); white-space:nowrap; }
tr.top td.name { color:var(--gold); font-weight:600; }
tr.picked td { background:#2d3a2a; } tr.picked td.name::after { content:"  \2190 your pick"; color:#9fd4a8; font-weight:400; }
.lanes { color:#9fd4a8; font-size:13px; }
.pick { font-size:15px; } .pick b { color:#9fd4a8; }
.pool-group { margin:4px 0 6px; } .pool-group .hdr { color:var(--dim); font-size:12px; }
.pool-group span.c { display:inline-block; margin:2px 4px 0 0; padding:1px 6px; border-radius:4px; background:#2b2d35; font-size:12px; }
.pool-group span.new { outline:1px solid var(--gold); }
.pip { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:2px; vertical-align:-1px; }
.W{background:#f3ecd2}.U{background:#4d8fd6}.B{background:#8a7f86}.R{background:#d6503c}.G{background:#3c9a5a}.C{background:#b9b9c0}
footer { color:var(--dim); font-size:12px; padding:0 16px 16px; }
.deck { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:14px 16px; }
.deck h2 { margin:0 0 4px; font-size:18px; } .deck .sub { color:var(--dim); margin-bottom:12px; }
.cols { display:flex; gap:10px; flex-wrap:wrap; }
.col { flex:1; min-width:150px; background:#1a1b20; border:1px solid var(--line); border-radius:6px; padding:6px 8px; }
.col h3 { margin:0 0 4px; font-size:12px; color:var(--dim); font-weight:600; }
.col div { font-size:13px; padding:1px 0; } .col .low { color:#e0a050; }
.bars { margin-top:4px; } .bar { display:grid; grid-template-columns: 150px 1fr 56px; gap:8px; align-items:center; margin:5px 0; font-size:13px; }
.bar .track { height:12px; background:#2b2d35; border-radius:6px; overflow:hidden; } .bar .fill { height:100%; background:#5b6478; }
.bar.you .fill { background:var(--gold); } .bar.you { font-weight:600; }
.note { color:var(--dim); font-size:12px; margin-top:8px; }
@media (max-width: 1100px) { main { grid-template-columns: 1fr; } }
</style></head><body>
<header>
  <h1 id="title"></h1>
  <button id="prev">&#9664; Prev</button><span id="where"></span><button id="next">Next &#9654;</button>
  <button id="toggle" class="on" title="T">Blacked out</button>
  <div id="bar"><div></div></div>
</header>
<main>
  <div class="shot" id="shot"></div>
  <aside>
    <div class="card"><h2>Pick</h2><div class="pick" id="pick"></div><div class="lanes" id="lanes"></div></div>
    <div class="card"><h2>Pack ranking</h2><table id="rank"></table></div>
    <div class="card"><h2 id="poolhdr">Your picks so far</h2><div id="pool"></div></div>
  </aside>
</main>
<footer>Left/Right arrows: previous/next pick &middot; T: blacked out / full screen &middot; Scores are the advisor's estimated win-rate gain.</footer>
<script>
const D = __DATA__;
let i = 0, blacked = true;
const N = D.steps.length + (D.deck ? 1 : 0);
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const COLORS = ["W","U","B","R","G"], CNAME = {W:"White",U:"Blue",B:"Black",R:"Red",G:"Green"};
$("title").textContent = D.event;
function group(name) {
  const c = D.cards[name];
  if (!c) return "Other";
  if (c.land) return "Lands";
  if (!c.colors) return "Colorless";
  if (c.colors.length > 1) return "Multicolor";
  return CNAME[c.colors];
}
function pips(name) {
  const c = D.cards[name]; if (!c) return "";
  const cols = c.land || !c.colors ? "C" : c.colors;
  return [...cols].map(x => `<span class="pip ${x}"></span>`).join("");
}
function deckHTML(dk, label) {
  const curveOrder = ["0","1","2","3","4","5","6+"];
  const byMv = {};
  dk.spells.forEach(e => { const k = e.mv >= 6 ? "6+" : String(Math.max(e.mv || 0, 0)); (byMv[k] = byMv[k] || []).push(e); });
  const cols = curveOrder.filter(k => byMv[k]).map(k =>
    `<div class="col"><h3>${k === "0" ? "0" : k} mana (${byMv[k].reduce((a, e) => a + e.count, 0)})</h3>` +
    byMv[k].map(e => `<div class="${e.castable < 0.8 ? "low" : ""}" title="castable ${Math.round(e.castable * 100)}%">` +
      `${e.count > 1 ? e.count + "x " : ""}${pips(e.name)} ${esc(e.name)}</div>`).join("") + `</div>`).join("") +
    `<div class="col"><h3>Lands (${dk.lands.reduce((a, e) => a + e.count, 0)})</h3>` +
    dk.lands.map(e => `<div>${e.count > 1 ? e.count + "x " : ""}${esc(e.name)}</div>`).join("") + `</div>`;
  let bars = "";
  const v = dk.vs_humans;
  if (v) {
    const rows = [["Average human deck", v.avg], ["Winning-record decks", v.avg_winning], ["7-win decks", v.avg_7_wins], ["Your deck", dk.win_rate]];
    const lo = Math.min(...rows.map(r => r[1])) - 0.05, hi = Math.max(...rows.map(r => r[1])) + 0.02;
    bars = `<div class="bars">` + rows.map(([n, x]) => `<div class="bar ${n === "Your deck" ? "you" : ""}"><span>${n}</span>` +
      `<div class="track"><div class="fill" style="width:${Math.max(3, 100 * (x - lo) / (hi - lo))}%"></div></div><span>${(100 * x).toFixed(1)}%</span></div>`).join("") +
      `</div><div class="note">Better than ${Math.round(v.percentile)}% of human decks and ${Math.round(v.percentile_vs_winning)}% of winning-record decks ` +
      `(${v.n} 17Lands Premier Draft pools, built and rated the same way). Judged win rate from an independent model with neutral player skill; ` +
      `compare decks, not players. Quick Draft bots pass stronger cards than human pods, so Quick Draft pools tend to rate higher.</div>`;
  }
  return `<div class="deck"><h2>${label}: ${esc(dk.lane)}</h2><div class="sub">${dk.spells.reduce((a, e) => a + e.count, 0)} spells, ` +
    `${dk.lands.reduce((a, e) => a + e.count, 0)} lands, ${dk.creatures} creatures &middot; spells castable on average ${Math.round(100 * dk.mean_castability)}% ` +
    `(orange = under 80%) &middot; judged win rate <b>${(100 * dk.win_rate).toFixed(1)}%</b></div><div class="cols">${cols}</div>${bars}</div>`;
}
function renderDeck() {
  $("where").textContent = `Deck  (${N}/${N})`;
  $("prev").disabled = false; $("next").disabled = true;
  $("bar").firstElementChild.style.width = "100%";
  $("toggle").style.visibility = "hidden";
  $("shot").innerHTML = deckHTML(D.deck.proposed, "Proposed deck") + (D.deck.human_style ? `<div style="height:12px"></div>` + deckHTML(D.deck.human_style, "Alternative (colors humans would most likely pick)") : "");
  $("shot").style.background = "transparent"; $("shot").style.border = "none";
  const s = D.steps[D.steps.length - 1];
  $("pick").innerHTML = `Your final pool: ${D.deck.pool_size} cards`;
  $("lanes").textContent = D.final_lanes ? "Final pool leans: " + D.final_lanes.map(([p, w]) => `${p} ${Math.round(100 * w)}%`).join("  ") : "";
  $("rank").innerHTML = `<tr><td>The deck uses the same builder and rating as the bot-vs-human comparison.</td></tr>`;
  const pool = s.pool.concat(s.pick ? [s.pick] : []);
  $("poolhdr").textContent = `All your picks (${pool.length})`;
  renderPool(pool, null);
}
function renderPool(pool, newest) {
  const groups = {};
  pool.forEach((n, k) => (groups[group(n)] = groups[group(n)] || []).push([n, newest && k === pool.length - 1]));
  const order = ["White","Blue","Black","Red","Green","Multicolor","Colorless","Lands","Other"];
  $("pool").innerHTML = order.filter(g => groups[g]).map(g =>
    `<div class="pool-group"><div class="hdr">${g} (${groups[g].length})</div>` +
    groups[g].map(([n, isNew]) => `<span class="c ${isNew ? "new" : ""}">${pips(n)} ${esc(n)}</span>`).join("") + `</div>`).join("");
}
function render() {
  if (D.deck && i === D.steps.length) return renderDeck();
  $("toggle").style.visibility = "visible";
  $("shot").style.background = ""; $("shot").style.border = "";
  const s = D.steps[i];
  $("where").textContent = `${s.slot}  (${i + 1}/${N})`;
  $("prev").disabled = i === 0; $("next").disabled = i === N - 1;
  $("bar").firstElementChild.style.width = `${100 * (i + 1) / N}%`;
  $("toggle").textContent = blacked ? "Blacked out" : "Full screen";
  $("toggle").classList.toggle("on", blacked);
  $("shot").innerHTML = s.img
    ? `<img src="${esc(blacked ? s.img.after : s.img.before)}" alt="${esc(s.slot)} screen"><span class="tag">${blacked ? "As you saw it" : "Full screen"}</span>`
    : `<div class="none">No screenshot for ${esc(s.slot)}${s.pack_size === 1 ? " (last card of the pack)" : ""}.</div>`;
  $("pick").innerHTML = s.pick
    ? `You took <b>${esc(s.pick)}</b>` + (s.bot_rank ? ` &mdash; advisor's #${s.bot_rank}` : "")
    : "No pick recorded";
  $("lanes").textContent = s.lanes.length ? "Projected deck: " + s.lanes.map(([p, w]) => `${p} ${w}%`).join("  ") : "";
  $("rank").innerHTML = s.ranking.map(r =>
    `<tr class="${r.top ? "top" : ""} ${r.name === s.pick ? "picked" : ""}"><td class="num">${r.rank}.</td>` +
    `<td class="name">${pips(r.name)} ${esc(r.name)}</td><td class="num">${r.score === null ? "--" : (r.score >= 0 ? "+" : "") + r.score.toFixed(2) + "%"}</td></tr>`).join("")
    || `<tr><td>No ranking recorded.</td></tr>`;
  const pool = s.pool.concat(s.pick ? [s.pick] : []);
  $("poolhdr").textContent = `Your picks so far (${pool.length})`;
  renderPool(pool, s.pick);
}
$("prev").onclick = () => { if (i > 0) { i--; render(); } };
$("next").onclick = () => { if (i < N - 1) { i++; render(); } };
$("toggle").onclick = () => { if (!(D.deck && i === D.steps.length)) { blacked = !blacked; render(); } };
document.addEventListener("keydown", e => {
  if (e.key === "ArrowRight") $("next").click();
  else if (e.key === "ArrowLeft") $("prev").click();
  else if (e.key.toLowerCase() === "t") $("toggle").click();
});
render();
</script></body></html>
"""

if __name__ == "__main__":
    sys.exit(main())
