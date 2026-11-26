"""Render a pick-by-pick draft report as a self-contained HTML page.

Each step shows the pool so far (by color, with a curve summary), the pack, and the
bot's ranking of every card in it, with the score split into card value and pool
synergy, plus the pick actually made.

Scores are win-logit gains from the deck model. They are shown as approximate
win-rate percentage points (logit x p(1-p), with p the format's average win rate).
"""

from __future__ import annotations

import html
from collections import Counter
from dataclasses import dataclass, field

WUBRG = "WUBRG"
COLOR_NAME = {"W": "White", "U": "Blue", "B": "Black", "R": "Red", "G": "Green", "M": "Multicolor", "C": "Colorless"}


@dataclass
class Step:
    slot: tuple[int, int]
    pool: list[int]                    # grpIds picked before this step
    pack: list[int] | None             # grpIds in the pack (None if not logged)
    ranking: list[dict] | None         # scorer.explain() output, best first
    pick: int | None = None
    pick_source: str = ""              # logged / inferred / bot / ""
    lanes: list | None = None          # deck-aware scorer: [(pair, weight, playables), ...]


@dataclass
class DraftReport:
    title: str
    subtitle: str
    steps: list[Step] = field(default_factory=list)
    base_win_rate: float = 0.545

    # ---- helpers --------------------------------------------------------------
    def pct(self, logit: float) -> float:
        p = self.base_win_rate
        return 100.0 * p * (1 - p) * logit


def _esc(s) -> str:
    return html.escape(str(s))


def _group(card) -> str:
    if card is None:
        return "C"
    if len(card.colors) > 1:
        return "M"
    return card.colors or "C"


def _mana(cost: str) -> str:
    if not cost:
        return ""
    out = []
    for sym in cost.strip("{}").split("}{"):
        cls = sym if sym in WUBRG else ("H" if "/" in sym else "N")
        out.append(f'<span class="pip p{cls}">{_esc(sym)}</span>')
    return "".join(out)


def _type_short(card) -> str:
    if card is None:
        return ""
    t = card.types
    for name in ("Creature", "Planeswalker", "Land", "Instant", "Sorcery", "Artifact", "Enchantment", "Battle"):
        if name in t:
            return name
    return "/".join(t)


def _lanes_html(lanes) -> str:
    if not lanes:
        return ""
    parts = " &middot; ".join(f'<b>{_esc(p)}</b> {100 * w:.0f}% <span class="muted">({n} playables)</span>'
                              for p, w, n in lanes)
    return f'<div class="lanes" title="Most likely color pairs for the final deck">Projected deck: {parts}</div>'


def _pool_html(pool: list[int], db) -> str:
    cards = [db.get(g) for g in pool]
    if not cards:
        return '<p class="muted">Empty: this is the first pick.</p>'
    nonland = [c for c in cards if c and "Land" not in c.types]
    color_count = Counter(col for c in nonland for col in c.colors)
    main = [col for col, _ in color_count.most_common(2)]
    creatures = sum(1 for c in nonland if "Creature" in c.types)
    curve = Counter(min(c.mana_value, 6) for c in nonland)
    curve_html = "".join(
        f'<div class="bar"><div class="fill" style="height:{4 + 8 * curve.get(mv, 0)}px"></div>'
        f'<span>{mv if mv < 6 else "6+"}</span><b>{curve.get(mv, 0) or ""}</b></div>'
        for mv in range(0, 7))
    pips = "".join(f'<span class="pip p{col}">{col}</span>{color_count[col]} ' for col in WUBRG if color_count[col])
    head = (f'<div class="poolhead"><div><b>{len(cards)}</b> cards · {creatures} creatures · '
            f'{len(nonland) - creatures} other spells · {len(cards) - len(nonland)} lands</div>'
            f'<div>Colors: {pips or "none"}</div>'
            f'<div>Leaning: <b>{"".join(main) or "-"}</b></div></div>'
            f'<div class="curve" title="Mana curve of nonland cards">{curve_html}</div>')
    groups: dict[str, list] = {}
    for g, c in zip(pool, cards):
        key = "L" if c and "Land" in c.types else _group(c)
        groups.setdefault(key, []).append(c if c else None)
    order = [*WUBRG, "M", "C", "L"]
    cols = []
    for key in order:
        if key not in groups:
            continue
        label = "Lands" if key == "L" else COLOR_NAME[key]
        items = sorted(groups[key], key=lambda c: (c.mana_value if c else 0, c.name if c else ""))
        lis = "".join(f'<li>{_esc(c.name)} <span class="cost">{_mana(c.mana_cost)}</span></li>' for c in items if c)
        cols.append(f'<div class="pcol"><h5 class="c{key}">{label} ({len(items)})</h5><ul>{lis}</ul></div>')
    return head + f'<div class="pcols">{"".join(cols)}</div>'


def _partners(items, rep: DraftReport, cls: str) -> str:
    return ", ".join(f'<span class="{cls}">{_esc(n)} {rep.pct(v):+.2f}</span>' for n, v in items)


def _ranking_html(step: Step, db, rep: DraftReport) -> str:
    pick_name = db.name(step.pick) if step.pick is not None else None
    rows = []
    rank_of_pick = None
    for i, r in enumerate(step.ranking, 1):
        g = next((g for g in step.pack if db.name(g) == r["name"]), None)
        c = db.get(g) if g is not None else None
        is_pick = r["name"] == pick_name
        if is_pick and rank_of_pick is None:
            rank_of_pick = i
        cls = " pickrow" if is_pick else ""
        cls += " top" if i == 1 else ""
        has_fit = "fit" in r
        if r["score"] != r["score"]:
            score = f'<td class="num muted" colspan="{4 if has_fit else 3}">{_esc(r["note"])}</td>'
        else:
            fit = f'<td class="num">{100 * r["fit"]:.0f}%</td>' if has_fit else ""
            score = (f'<td class="num strong">{rep.pct(r["score"]):+.2f}</td>{fit}'
                     f'<td class="num">{rep.pct(r["value"]):+.2f}</td>'
                     f'<td class="num">{rep.pct(r["synergy"]):+.2f}</td>')
        partners = _partners(r["best"], rep, "pos") + ("; " if r["best"] and r["worst"] else "") + _partners(r["worst"], rep, "neg")
        tag = ' <span class="tag you">your pick</span>' if is_pick and step.pick_source in ("logged", "inferred") else \
              (' <span class="tag bot">bot pick</span>' if is_pick else "")
        rows.append(
            f'<tr class="{cls.strip()}"><td class="num">{i}</td>'
            f'<td><span class="sw c{_group(c)}"></span>{_esc(r["name"])}{tag}</td>'
            f'<td class="cost">{_mana(c.mana_cost) if c else ""}</td>'
            f'<td class="muted">{_esc(_type_short(c))}</td><td class="muted r{c.rarity[:1] if c else ""}">{_esc(c.rarity if c else "")}</td>'
            f'{score}<td class="partners">{partners}</td></tr>')
    has_fit = any("fit" in r for r in step.ranking)
    fit_th = ('<th class="num" title="Share of the likely decks (by color pair weight) that can play this card">'
              'Fit</th>') if has_fit else ""
    table = ('<table><thead><tr><th>#</th><th>Card</th><th>Cost</th><th>Type</th><th>Rarity</th>'
             '<th class="num" title="Ranking score: estimated win-rate gain to the likely final deck">Score</th>'
             + fit_th +
             '<th class="num" title="Card on its own">Value</th>'
             '<th class="num" title="Synergy with the pool so far">Synergy</th>'
             '<th>Best / worst pairs in pool</th></tr></thead><tbody>' + "".join(rows) + "</tbody></table>")
    return table, rank_of_pick


def render(rep: DraftReport, db) -> str:
    sections, nav = [], []
    agree = compared = 0
    misses = []
    for idx, st in enumerate(rep.steps):
        anchor = f"p{st.slot[0]}p{st.slot[1]}"
        label = f"P{st.slot[0]}P{st.slot[1]}"
        pick_name = db.name(st.pick) if st.pick is not None else None
        src_txt = {"logged": "your pick", "inferred": "your pick, inferred from the final pool",
                   "bot": "made by the bot (your pick was not logged)", "": "unknown"}[st.pick_source]
        if st.ranking is None:
            body = '<p class="muted">This pack was not in the log (Arena restarted).</p>'
            verdict = f'Pick: <b>{_esc(pick_name or "?")}</b> ({src_txt}).'
            nav_cls = "nopack"
        else:
            table, rank = _ranking_html(st, db, rep)
            top = st.ranking[0]
            n = len(st.ranking)
            if pick_name is None:
                verdict = f'Bot\'s pick: <b>{_esc(top["name"])}</b>. Your pick is unknown.'
                nav_cls = "unknown"
            elif st.pick_source == "bot":
                verdict = f'Bot picked <b>{_esc(pick_name)}</b>.'
                nav_cls = "bot"
            else:
                match = rank == 1
                if n > 1:
                    compared += 1
                    agree += match
                if not match and rank is not None:
                    pick_score = next(r["score"] for r in st.ranking if r["name"] == pick_name)
                    gap = top["score"] - pick_score if pick_score == pick_score else float("nan")
                    misses.append((gap, label, anchor, top["name"], pick_name, rank, n))
                verdict = (f'{"&#10003; Matches the bot" if match else "Bot preferred <b>" + _esc(top["name"]) + "</b>"}'
                           f' &mdash; you took <b>{_esc(pick_name)}</b> ({src_txt}), bot rank #{rank} of {n}.')
                nav_cls = "match" if match else "miss"
            body = table
        nav.append(f'<a class="{nav_cls}" href="#{anchor}" title="{_esc(pick_name or "")}">{label}</a>')
        pack_size = len(st.pack) if st.pack else 0
        sections.append(
            f'<section id="{anchor}"><h3><a class="totop" href="#">top</a>{label} <span class="muted">· pick {idx + 1} · '
            f'{pack_size or "?"} cards in pack</span></h3><p class="verdict">{verdict}</p>'
            f'<div class="grid"><div class="pool"><h4>Pool before this pick</h4>{_lanes_html(st.lanes)}{_pool_html(st.pool, db)}</div>'
            f'<div class="pack"><h4>Pack, ranked by the bot</h4>{body}</div></div></section>')

    final_pool = [s.pick for s in rep.steps if s.pick is not None]
    misses.sort(key=lambda m: -(m[0] if m[0] == m[0] else -1))
    miss_rows = "".join(
        f'<tr><td><a href="#{a}">{lbl}</a></td><td>{_esc(top)}</td><td>{_esc(you)}</td>'
        f'<td class="num">#{r} of {n}</td><td class="num">{rep.pct(gap):+.2f}</td></tr>'
        for gap, lbl, a, top, you, r, n in misses[:8])
    summary = (
        f'<section id="summary"><h3><a class="totop" href="#">top</a>Summary</h3>'
        f'<p>The bot\'s top card matched the real pick on <b>{agree} of {compared}</b> picks '
        f'({(100 * agree / compared) if compared else 0:.0f}%), not counting single-card packs. '
        f'For reference, offline across thousands of drafts the bot matches human picks about 24% of the time.</p>'
        + (f'<h4>Biggest disagreements</h4><table><thead><tr><th>Pick</th><th>Bot\'s choice</th><th>Pick made</th>'
           f'<th class="num">Bot rank of pick</th><th class="num">Score gap</th></tr></thead><tbody>{miss_rows}</tbody></table>'
           if miss_rows else "")
        + f'<h4>Final pool</h4>{_pool_html(final_pool, db)}</section>')

    legend = ('<p class="muted legend">Scores are the deck model\'s estimate of how much each card would change the win '
              'rate of the deck being drafted, in percentage points. With the deck-aware scorer, <b>Score</b> is the '
              'gain to the decks the pool is likely to become (weighted over color pairs) and <b>Fit</b> is the share '
              'of those likely decks that could play the card. <b>Value</b> is the card on its own; '
              '<b>Synergy</b> is its fit with all cards already picked. Basic lands are never recommended. '
              'Nav colors: <span class="key match">matches bot</span> <span class="key miss">differs</span> '
              '<span class="key bot">bot picked</span> <span class="key unknown">pick unknown</span> '
              '<span class="key nopack">pack missing</span>. Data: 17Lands (CC BY 4.0).</p>')
    return PAGE.format(title=_esc(rep.title), subtitle=_esc(rep.subtitle), nav="".join(nav),
                       legend=legend, summary=summary, sections="".join(sections))


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {{ --bg:#fbfaf8; --fg:#1d1d1f; --muted:#6b6b70; --line:#e4e2dd; --card:#ffffff; --hl:#fff4d6; --top:#eef6ee;
  --pos:#1d7a3a; --neg:#b3261e; --accent:#3b5bdb; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#17171a; --fg:#ececef; --muted:#9a9aa2; --line:#2c2c31; --card:#1f1f23;
  --hl:#3a3220; --top:#1f3324; --pos:#6fcf8f; --neg:#ff8a80; --accent:#8fa8ff; }} }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--fg); font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif; }}
header {{ background:var(--bg); border-bottom:1px solid var(--line); padding:12px 24px 10px; }}
@media (min-width:1100px) and (min-height:700px) {{ header {{ position:sticky; top:0; z-index:2; }} }}
.totop {{ float:right; font-size:12px; font-weight:400; color:var(--accent); text-decoration:none; }}
h1 {{ font-size:20px; margin:0 0 2px; }} h3 {{ font-size:17px; margin:0 0 4px; }} h4 {{ font-size:13px; margin:0 0 8px; text-transform:uppercase; letter-spacing:.04em; color:var(--muted); }}
h5 {{ font-size:12px; margin:0 0 4px; }}
.sub {{ color:var(--muted); margin:0 0 8px; }}
nav {{ display:flex; flex-wrap:wrap; gap:4px; }}
nav a, .key {{ font-size:11px; padding:2px 6px; border-radius:4px; text-decoration:none; color:var(--fg); border:1px solid var(--line); }}
.match {{ background:#c9ebd2; color:#123d1e !important; }} .miss {{ background:#f6d5d1; color:#4a1410 !important; }}
.bot {{ background:#d9e0fb; color:#16245a !important; }} .unknown {{ background:transparent; }} .nopack {{ background:#e9e6df; color:#555 !important; }}
main {{ padding:8px 24px 48px; max-width:1500px; }}
section {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:16px 18px; margin:16px 0; scroll-margin-top:110px; }}
.verdict {{ margin:2px 0 12px; }}
.grid {{ display:grid; grid-template-columns:minmax(260px, 1fr) 2.6fr; gap:22px; }}
@media (max-width:1000px) {{ .grid {{ grid-template-columns:1fr; }} }}
table {{ border-collapse:collapse; width:100%; font-size:13px; }}
th, td {{ text-align:left; padding:4px 8px; border-bottom:1px solid var(--line); vertical-align:top; }}
th {{ font-size:11px; color:var(--muted); font-weight:600; }}
.num {{ text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }}
.strong {{ font-weight:650; }} .muted {{ color:var(--muted); }}
tr.top td {{ background:var(--top); }} tr.pickrow td {{ background:var(--hl); }}
.partners {{ font-size:12px; color:var(--muted); }} .pos {{ color:var(--pos); }} .neg {{ color:var(--neg); }}
.tag {{ font-size:10px; font-weight:700; padding:1px 5px; border-radius:3px; margin-left:4px; white-space:nowrap; }}
.tag.you {{ background:#f2c14e; color:#3a2a00; }} .tag.bot {{ background:#9fb3f5; color:#101c48; }}
.pip {{ display:inline-block; min-width:16px; height:16px; line-height:16px; border-radius:8px; text-align:center; font-size:10px; font-weight:700; margin-right:1px; color:#222; }}
.pW {{ background:#f3eccf; }} .pU {{ background:#a9d0f0; }} .pB {{ background:#bdb3ad; }} .pR {{ background:#f1a487; }} .pG {{ background:#9fd4a8; }} .pN, .pH {{ background:#d6d3cc; }}
.sw {{ display:inline-block; width:8px; height:8px; border-radius:2px; margin-right:6px; vertical-align:middle; }}
.cW {{ background:#e8dcae; }} .cU {{ background:#5a9fd6; }} .cB {{ background:#6d625c; }} .cR {{ background:#d9603b; }} .cG {{ background:#4a9a5b; }} .cM {{ background:#c9a227; }} .cC, .cL {{ background:#aaa59b; }}
h5.cW, h5.cU, h5.cB, h5.cR, h5.cG, h5.cM, h5.cC, h5.cL {{ background:none; }}
.rr, .rm {{ color:#b8860b; }}
.poolhead {{ display:grid; gap:2px; font-size:13px; margin-bottom:8px; }}
.curve {{ display:flex; gap:6px; align-items:flex-end; height:70px; margin:6px 0 10px; }}
.bar {{ display:flex; flex-direction:column; align-items:center; justify-content:flex-end; width:22px; font-size:10px; color:var(--muted); }}
.bar .fill {{ width:16px; background:var(--accent); border-radius:3px 3px 0 0; opacity:.75; }}
.bar b {{ color:var(--fg); font-weight:600; height:12px; order:-1; }}
.pcols {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(150px, 1fr)); gap:8px 14px; }}
.pcol ul {{ list-style:none; padding:0; margin:0; font-size:12px; }} .pcol li {{ padding:1px 0; }}
.cost {{ white-space:nowrap; }}
.legend {{ font-size:12px; margin:6px 0 0; }}
.lanes {{ font-size:13px; margin:0 0 8px; padding:6px 8px; border-radius:6px; background:var(--top); }}
.legend .key {{ margin-right:2px; }}
</style></head>
<body><header><h1>{title}</h1><p class="sub">{subtitle}</p><nav>{nav}<a href="#summary">Summary</a></nav>{legend}</header>
<main>{sections}{summary}</main></body></html>
"""
