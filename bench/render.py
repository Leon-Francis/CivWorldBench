"""Benchmark 的渲染层 —— 报告格式、预热段 H、外生动作 A 的**单一真源**。

★ 为什么这是一个独立模块，而不是留在 `tools/gen_bench_examples.py` 里：
**报告格式是 benchmark 的身份，要冻结进 `bench/manifest.json` 的 digest。**
挂在一个文档生成脚本里的话，改文档排版就会静默改掉数据集。
`tools/gen_bench_examples.py`（文档实例）与 `bench/build_bench.py`（出题）现在都从这里 import。

⚠️ 同理，这里**刻意不放进** `bundle/sandboxes/freeciv/report.py`：那边的 `full`/`compact`
要跟着语料演进，两者生命周期不同，绑一起必然互相拖累。地图段与外交段仍然**整段原样**
取自 `report.py` 的 compact 渲染（见 `map_section`），因为 `verify_compact.py` 的字节稳定
闸门只覆盖 `report.py`，自己重画一定漂移。

⚠️ 不要"顺手修" `proto_stale_obs.owner_name`：`dc.owner` 与当前真值不一致**正是记忆该有的
行为**，把它改成当前真值会让视角 2 静默退化成视角 1，而从输出上完全看不出来。

不编造任何内容：报告与 events 来自 corpus，真值来自 branch 的 `factual`/`resolved`
与 `probes.A_static`，视角 2 的观测来自 `probes.C_belief[viewer].dc` + 自算当前视野。
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bundle"))
sys.path.insert(0, str(ROOT / "tools"))
from sandboxes.freeciv.state import base_ruleset  # noqa: E402
from proto_stale_obs import vision_mask, owner_name  # noqa
from bench_report import (civ_block, unit_counts, city_rows, CITY_HDR,  # noqa
                          unit_is_military, interval_narrative,
                          resource_stats, _fmt_counter)
# ⚠️ branch 记录里序列化的 `op` 字典**没有** score_vars（只有 id/layer/target/applied/
# expected_path/horizon_band/notes）。它定义在 interventions.LIBRARY 里，必须从那里取，
# 否则表里会印出一个空的 score_vars 而看起来像"这个算子没有评分变量"。
from sandboxes.freeciv.interventions import LIBRARY as _OPS  # noqa
# ★ 事件渲染复用 report._fmt_event —— 单一真源。自己再写一份就会与语料的词表漂移，
# 而那正是旧事件流三类 bug 的成因之一。
from sandboxes.freeciv.report import _fmt_event, _DIPL_PHRASE  # noqa
from sandboxes.freeciv.event_stream import load as load_event_stream  # noqa
SCORE_VARS = {o.id: tuple(o.score_vars) for o in _OPS}

SCALARS = ("gold", "population", "score", "culture", "government", "n_cities", "n_techs",
           "n_units", "at_war", "units_lost", "units_killed", "units_built",
           "n_wonders", "explored", "alive")

# A 陈述的量必须从评分里排除 —— 与任务 3 只在算子 score_vars 上评分是同一套机制
# ⚠️ 按 2026-09-02 修正后的词表。`peace_treaty` 已删除（它是错的）；
# 所有外交类型都影响 `at_war`，所以 A 里出现任何一条就要排除它。
A_EXCLUDES = {"government_change": ("government",), "tech_discovered": ("n_techs",),
              "city_founded": ("n_cities",), "city_destroyed": ("n_cities",),
              "city_transferred": ("n_cities",), "wonder_built": ("n_wonders",)}
A_EXCLUDES.update({t: ("at_war",) for t in _DIPL_PHRASE})


def strip_map(rep: str) -> str:
    """去掉 THE MAP 段 —— H 的非首帧不重复给地图（地形静态，占 47% token）。"""
    return re.sub(r'\n\n#{2,3} THE MAP.*?(?=\n\n#{2,3} )', '', rep, flags=re.S)


def strip_events(rep: str) -> str:
    """去掉 EVENTS 段 —— events 已是输入侧的动作通道，不再是被预测的输出。"""
    return re.sub(r'\n\n#{2,3} EVENTS THIS INTERVAL.*?(?=\n\n#{2,3} |\Z)', '', rep, flags=re.S)


def group_events(evs: list[dict]) -> list[str]:
    """同一回合的事件并到一行，省空间。

    分组键用 `engine_turn`（引擎自己的记录，精确）优先、退回 `turn`（状态 diff 首次可见
    的回合，只承诺区间）—— 因为实测 diff 的 turn 对 city_founded 30/30、
    tech_discovered 57/58、government_change 14/16 都晚一回合，而偏移逐类型不同。
    ⇒ 两种来源不能混在同一个"turn N"标签下，否则精确的和只承诺区间的看起来一样。
    """
    import collections
    # 不区分精确/区间 —— 一律用 engine_turn 优先、退回 turn 的那个数当回合号。
    # （精度差异记在文档的已知边界里，正文不再每行提醒。）
    by = collections.defaultdict(list)
    for e in evs:
        et = e.get("engine_turn")
        by[et if et is not None else e.get("turn")].append(e)
    return [f"  turn {t}: " + "; ".join(ev_phrase(e) for e in by[t])
            for t in sorted(by, key=lambda x: x or 0)]


def ev_phrase(e: dict) -> str:
    """事件的短语部分（不含回合前缀）—— 供 group_events 拼接。"""
    txt = ev_line(e).strip()
    for pref in ("turn ", "by turn "):
        if txt.startswith(pref):
            i = txt.find(": ")
            if i > 0:
                return txt[i + 2:]
    return txt


def ev_line(e: dict) -> str:
    """一行一条事件。渲染委托给 `report._fmt_event` —— 单一真源，词表随语料演进。

    自己再写一份就会与语料词表漂移，而那正是旧事件流三类 bug 的成因之一
    （`report.py` 里硬编码 `("war_declared","peace_treaty")` 的视角过滤同理）。
    """
    if e["type"] == "wonder_built":            # 派生类型，report 词表里没有
        where = f" in {e['city']}" if e.get("city") else ""
        return f"  turn {e['turn']}: {e['player']} completed wonder {e['wonder']}{where}"
    return "  " + _fmt_event(e)


def derive_wonders(recs: dict) -> list[dict]:
    """★ 语料没有 wonder_built 事件类型 —— 从帧间 small_wonders/wonders 的 diff 派生。"""
    out, prev = [], {}
    for t in sorted(recs):
        for p in recs[t]["state"]["players"]:
            cur = set(p.get("wonders") or []) | set(p.get("small_wonders") or [])
            for w in cur - prev.get(p["name"], set()):
                out.append({"type": "wonder_built", "player": p["name"],
                            "wonder": w, "city": "", "turn": t})
            prev[p["name"]] = cur
    return out


def visible_to(e: dict, viewer: str, rec: dict) -> bool:
    """★ 视角 2 的 events 知情过滤：引擎的真实视角下该 civ 知道什么就给什么。"""
    actor = e.get("player")
    if actor == viewer:
        return True
    if actor is None:
        # v2 engine events whose actor was consumed by the action (a `finished`
        # record after Steal Tech / Found City ...) carry no civ; the matching
        # `started` event does, so nothing is lost by hiding these in view 2.
        return viewer in (e.get("other"), e.get("from_player"))
    cb = rec["probes"]["C_belief"][viewer]
    ds = (cb.get("diplstate_hidden") or {})
    def met(o):   return (ds.get(o) or {}).get("closest", "Never met") != "Never met"
    def emb(o):   return bool((ds.get(o) or {}).get("embassy"))
    if e["type"] in _DIPL_PHRASE:               # 全部外交类型
        if viewer in (actor, e.get("other")):
            return True                         # 自己是当事方
        return met(actor) or met(e.get("other", ""))
    if e["type"] in ("tech_discovered", "government_change", "wonder_built"):
        return emb(actor)                       # 内政，需要大使馆
    if e["type"] == "city_transferred":
        # 易手：两方任一是自己就看得见；否则要见过该地块
        if viewer in (actor, e.get("from_player")):
            return True
        return _city_tile_known(e, viewer, rec)
    if e["type"] in ("city_founded", "city_destroyed"):
        return met(actor) and _city_tile_known(e, viewer, rec)
    return met(actor)


def _city_tile_known(e: dict, viewer: str, rec: dict) -> bool:
    kn = rec["geo"]["grids"]["known"][viewer]
    for p in rec["state"]["players"]:
        for c in p.get("cities") or []:
            if c["name"] == e.get("city"):
                row = kn[c["y"]] if c["y"] < len(kn) else ""
                return c["x"] < len(row) and str(row[c["x"]]) not in "0False"
    return False


def render_stale_report(rec: dict, viewer: str, with_map: bool, ruleset: str = "") -> str:
    """视角 2 的观测：自己完整 + 外国城市用 dc 记忆 + 外国单位只在当前视野内。"""
    st, geo, cb = rec["state"], rec["geo"], rec["probes"]["C_belief"][viewer]
    X, Y = geo["xsize"], geo["ysize"]
    me = next(p for p in st["players"] if p["name"] == viewer)
    vis = vision_mask(me, X, Y)
    known = geo["grids"]["known"][viewer]
    mil = unit_is_military(ruleset or "civ2civ3")
    terr = rec["geo"].get("territory") or {}
    _, res_civ, res_city = resource_stats(rec)
    L = [f"## WORLD REPORT (as {viewer} believes it)",
         f"turn {rec['t']}, year {rec['year']}", f"ruleset {base_ruleset(rec['ruleset'])}", ""]
    if with_map:                       # ★ 段序与视角 1 统一：地图在最前
        X, Y = rec["geo"]["xsize"], rec["geo"]["ysize"]
        L += [f"### THE MAP  ({X} wide x {Y} tall)", "terrain; '?' = never explored"]
        for y, row in enumerate(rec["geo"]["grids"].get("terrain") or []):
            kr = known[y] if y < len(known) else ""
            L.append("".join(ch if (x < len(kr) and str(kr[x]) not in "0False") else "?"
                             for x, ch in enumerate(row)))
        L.append("")
    L += ["### CIVILIZATIONS", "-- mine --"] + civ_block(with_cost(me, rec, ruleset), terr, mil, res_civ, res_city)
    # ★ 外国文明：只给能观察到的。绝不给 gold / techs / score / trade —— 那是内政。
    L += ["", "-- foreign (what I can observe) --"]
    import collections
    # ★ 2026-09-15 (found by the Claude review run): `diplstate_hidden[..].closest` is the CLOSEST
    # relation ever reached (freeciv's max_state), not the current one -- the probe drops `current`
    # on purpose because "the report already prints it".  So this section showed "Alliance" for a
    # civ that was conquering the viewer's cities.  The viewer knows its own current relations:
    # take them from the state's diplomacy table (slot -> current state).
    slot_name = {f"player{i}": q["name"] for i, q in enumerate(st["players"])}
    current = {slot_name[k]: v for k, v in (me.get("diplomacy") or {}).items() if k in slot_name}
    for p in st["players"]:
        if p["name"] == viewer or not p.get("is_alive"):
            continue
        ds = (cb.get("diplstate_hidden") or {}).get(p["name"]) or {}
        if ds.get("closest", "Never met") == "Never met":
            L.append(f"{p['name']}: never met")
            continue
        remembered = [e for e in cb["dc"]
                      if owner_name(e["owner"], st["players"]) == p["name"]]
        seen = [u for u in (p.get("units") or []) if (u["x"], u["y"]) in vis]
        c = collections.Counter(u["type"] for u in seen)
        m = sum(v for k, v in c.items() if mil.get(k, False))
        L.append(f"{p['name']} of the {p['nation']}")
        L.append(f"  relation: {current.get(p['name'], ds.get('closest', '?'))}"
                 + ("; embassy" if ds.get("embassy") else "; no embassy"))
        L.append(f"  cities I remember ({len(remembered)}):")
        L.append("    city | x,y | size(remembered) | walls | capital")
        for e in sorted(remembered, key=lambda z: (z["y"], z["x"])):
            L.append(f"    {e['name']} | {e['x']},{e['y']} | {e['size']} | "
                     f"{'yes' if e.get('walls') else 'no'} | {e.get('capital','Not')}")
        L.append(f"  units in sight ({len(seen)}): "
                 + (", ".join(f"{k} x{v}" for k, v in c.most_common()) or "(none)")
                 + f"    military {m}, civilian {len(seen)-m}")
        if ds.get("embassy"):
            # 有大使馆才看得到内政的一部分（freeciv 的规则）
            L.append(f"  (embassy) treasury {p['gold']} gold, "
                     f"technologies {len(p['techs'])}, government {p['government']}")
    # 外交段放最后 —— 段序与视角 1 统一（THE MAP → CIVILIZATIONS → DIPLOMATIC）
    L += ["", "### DIPLOMATIC RELATIONS (as known to me)"]
    alive_names = {q["name"] for q in st["players"] if q.get("is_alive")}
    for other, ds in (cb.get("diplstate_hidden") or {}).items():
        if other == viewer or other not in alive_names:          # destroyed players: no relation to report
            continue
        L.append(f"- {other}: {current.get(other, ds.get('closest', '?'))}"
                 + ("; embassy" if ds.get("embassy") else "; no embassy")
                 + (f"; first contact t{ds['first_contact_turn']}"
                    if ds.get("first_contact_turn") else ""))
    return "\n".join(L)


def pomdp_known(rec: dict, viewer: str, civs: list[str], mil: dict) -> dict:
    """POMDP reference forecast (2026-09-10): the 1a values the viewer can *read off its own
    stale report* at t1 -- the honest persistence baseline for view 2.  Own civ: all 11 true
    values.  A met foreign civ: n_cities / city_size_total from the remembered-cities table,
    n_military_units / n_civilian_units = units currently in sight (a lower bound), gold and
    n_techs only with an embassy; everything else None (the scorer then falls back to
    climatology for that cell).  Mirrors render_stale_report exactly -- change both together."""
    import collections
    from bench.spec_v2 import derive
    st, geo = rec["state"], rec["geo"]
    cb = rec["probes"]["C_belief"][viewer]
    me = next(p for p in st["players"] if p["name"] == viewer)
    vis = vision_mask(me, geo["xsize"], geo["ysize"])
    out = {}
    for p in st["players"]:
        if p["name"] not in civs:
            continue
        if p["name"] == viewer:
            out[p["name"]] = derive(p, mil, rec); continue
        ds = (cb.get("diplstate_hidden") or {}).get(p["name"]) or {}
        remembered = [e for e in cb["dc"] if owner_name(e["owner"], st["players"]) == p["name"]]
        seen = [u for u in (p.get("units") or []) if (u["x"], u["y"]) in vis]
        m = sum(1 for u in seen if mil.get(u["type"], False))
        d = {"n_cities": len(remembered), "city_size_total": sum(int(e.get("size") or 0) for e in remembered),
             "n_military_units": m, "n_civilian_units": len(seen) - m}
        if ds.get("embassy"):
            d["gold"] = int(p["gold"]); d["n_techs"] = len(p.get("techs") or [])
        out[p["name"]] = d
    return out


def activity_digest(rec_a: dict, rec_b: dict, viewer: str, view: int,
                    with_production: bool = True) -> list[str]:
    """区间行为摘要 —— 状态 diff，不是事件。

    ★ 为什么必须有：离散事件太稀疏，而 `trait` 是**速率**，稀疏事件估不出速率。
    实测同一区间（turn 37..56）事件 0 条，而摘要能看出「Crazy Horse 单位 4→16 全是军事」
    「T'aejong 疆域 32→74」—— 这才是人格的证据。

    ★ 视角 2 只给**可观测的**量。别人的 `techs` / `units` / `gold` 是内政，视角文明
    看不到 —— 此前这里是个占位分支，摘要照样把所有文明的完整计数发出去，等于把视角 2
    的遮罩在摘要里全部漏回来。现在按视角分两条路径：

      自己          全部（城/单位/军事/科技/疆域/在造）
      别人 · 已接触  只有能观察到的：城数（记忆）、疆域、视野内单位数
      别人 · 有大使馆 追加科技数（freeciv 的规则：大使馆能看到部分内政）
      别人 · 未接触  完全不出现

    ⚠️ 摘要只覆盖**历史段**。跨到预测段就是泄漏，切分点由调用方保证。
    ⚠️ `building:` 那一列是 `actions` 城市生产字段的区间聚合 —— 报告只给当前帧在造什么，
    聚合是新信息，且是「造兵 vs 造建筑」最直接的 trait 证据。但它确实把我们刻意弃用的
    `actions` 的一部分放回来了。`with_production=False` 可关掉，两种设定都该跑一遍。
    """
    import collections
    out = []
    A = {p["name"]: p for p in rec_a["state"]["players"]}
    B = {p["name"]: p for p in rec_b["state"]["players"]}
    ta = rec_a["geo"].get("territory") or {}
    tb = rec_b["geo"].get("territory") or {}
    cb_a = (rec_a["probes"]["C_belief"].get(viewer) or {}) if view == 2 else {}
    cb_b = (rec_b["probes"]["C_belief"].get(viewer) or {}) if view == 2 else {}
    vis_b = vision_mask(B[viewer], rec_b["geo"]["xsize"], rec_b["geo"]["ysize"]) \
        if view == 2 and viewer in B else set()

    def remembered(cb, name, players):
        return sum(1 for e in (cb.get("dc") or [])
                   if owner_name(e["owner"], players) == name)

    for name in B:
        a, b = A.get(name), B[name]
        if a is None or not b.get("is_alive"):
            continue
        own = (name == viewer) or view == 1
        if not own:
            ds = ((cb_b.get("diplstate_hidden") or {}).get(name) or {})
            if ds.get("closest", "Never met") == "Never met":
                continue                       # 未接触 ⇒ 完全不出现
            ca = remembered(cb_a, name, rec_a["state"]["players"])
            cc = remembered(cb_b, name, rec_b["state"]["players"])
            seen = [u for u in (b.get("units") or []) if (u["x"], u["y"]) in vis_b]
            line = (f"  {name:<16} cities I remember {ca}->{cc}   "
                    f"units in sight {len(seen)}   "
                    f"territory {ta.get(name,{}).get('tiles','?')}->"
                    f"{tb.get(name,{}).get('tiles','?')}")
            if ds.get("embassy"):
                line += f"   (embassy) techs {len(a['techs'])}->{len(b['techs'])}"
            out.append(line)
            continue
        out.append(f"  {name:<16} cities {len(a['cities'])}->{len(b['cities'])}   "
                   f"units {len(a['units'])}->{len(b['units'])} "
                   f"(military {a.get('military_units',0)}->{b.get('military_units',0)})   "
                   f"techs {len(a['techs'])}->{len(b['techs'])}   "
                   f"territory {ta.get(name,{}).get('tiles','?')}->"
                   f"{tb.get(name,{}).get('tiles','?')}")
        if with_production:
            prod = collections.Counter(c["production"] for c in b["cities"])
            top = ", ".join(f"{k}x{v}" for k, v in prod.most_common(3))
            if top:
                out.append(f"  {'':<16} building: {top}")
    return out


def ruleset_primer(ruleset: str) -> str:
    """静态规则前言 —— 和地图一样，只在 H 开头给一次。

    ★ 为什么必须有：没有它，「模型答错」分不清是**不会推演**还是**不知道 Alphabet 解锁什么**，
    而那个混杂会污染整个 benchmark 的归因。primer 由 `sandboxes/freeciv/ruleset.py` 从
    freeciv 的 ruleset 文件生成，含科技树前置、单位、建筑、地形、政体五段。
    ⚠️ **两个 ruleset 都要用** —— test-ood 的 100 个 XL 世界是 `classic`，只给 civ2civ3
    的话，OOD 上的归因混杂原封不动地存在，而 OOD 恰恰最需要干净归因。
    """
    f = ROOT / f"bundle/docs/freeciv/ruleset_primer/RULES_{ruleset}.md"
    if not f.exists():
        raise SystemExit(f"缺 primer: {f}")
    return f.read_text(encoding="utf-8").rstrip()


def map_section(rec: dict) -> str:
    """`### THE MAP` 整段，原样取自 compact 报告。"""
    from sandboxes.freeciv.rehydrate import record_to_objects
    from sandboxes.freeciv.report import render
    ws, mi, _ = record_to_objects(rec)
    rep = render(ws, "compact", ws.players[0].name if ws.players else None, mi)
    m = re.search(r'#{2,3} THE MAP.*?(?=\n\n#{2,3} |\Z)', rep, re.S)
    if not m:
        raise SystemExit("compact 报告里找不到 THE MAP 段")
    sec = m.group(0).rstrip()
    # ★ 去掉逐地块的行式清单（`y=0  11:i  12:i`）—— 可读性差且长；它的内容改由报告里的
    # 资源统计承载。地形网格保留（它是密的，网格形式本来就合适）。
    # ⚠️ 曾试过把 tile detail 改成「每层一张网格」，实测**更大**：extras 稀疏
    # （3,840 格里仅 1,250 格有内容），网格要为每个空格也花一个字符。所以是聚合而非网格。
    sec = re.sub(r"\ntiles with resources or improvements.*?\Z", "", sec, flags=re.S)
    return sec.rstrip()


def dipl_section(rec: dict) -> str:
    """`### DIPLOMATIC RELATIONS` —— 各国之间的关系矩阵，同样取自 compact。"""
    from sandboxes.freeciv.rehydrate import record_to_objects
    from sandboxes.freeciv.report import render
    ws, mi, _ = record_to_objects(rec)
    rep = render(ws, "compact", ws.players[0].name if ws.players else None, mi)
    m = re.search(r'#{2,3} DIPLOMATIC RELATIONS.*?(?=\n\n#{2,3} |\Z)', rep, re.S)
    return m.group(0).rstrip() if m else "### DIPLOMATIC RELATIONS\n(none)"


def with_cost(pl: dict, rec: dict, ruleset: str) -> dict:
    """Attach the research cost of the civ's current tech (bench.techcost) when the
    caller stashed the world config on the record (`rec["_world"]`: sciencebox, ailevel)."""
    w = rec.get("_world")
    if not w or not pl.get("research_now") or pl.get("research_now") in ("A_FUTURE", "Future", "Future Tech"):
        return pl                                 # Future Technology repeats: cost formula not modelled (review run 2026-09-15)
    from bench.techcost import research_cost
    d = dict(pl)
    try:
        d["_cost"] = research_cost(rec, pl["name"], pl["research_now"], ruleset or "civ2civ3",
                                   int((w.get("params") or {}).get("sciencebox", 100)), w.get("ailevel", "hard"))
    except Exception:
        pass
    return d


def stat_report(rec: dict, ruleset: str, with_map: bool) -> str:
    """视角 1 的统计式报告：THE MAP 不变，其余按文明聚合成一块。"""
    st, geo = rec["state"], rec["geo"]
    mil = unit_is_military(ruleset or "civ2civ3")
    terr = geo.get("territory") or {}
    L = ["## WORLD REPORT", f"turn {rec['t']}, year {rec['year']}",
         f"ruleset {base_ruleset(rec['ruleset'])}", ""]   # ★ 无 world_id；v2 字段含 _ill 变体，打印基础名
    if with_map:
        # ★ 地图段**整段原样**取自语料的 compact 渲染器，不自己重画 —— 那一段含地形网格、
        # 行分组的 tile detail（`y=34  12:riW w123 s45` + 代码图例）、以及各层网格，
        # 自己重画一定会漂移，而且 `verify_compact.py` 的字节稳定闸门只覆盖 report.py。
        L.append(map_section(rec))
        L.append("")
    # 全局资源段去掉 —— 逐文明的已经够了，全局那行没有用
    _, res_civ, res_city = resource_stats(rec)
    L.append("### CIVILIZATIONS")
    for pl in st["players"]:
        if not pl.get("is_alive"):
            continue
        L += civ_block(with_cost(pl, rec, ruleset), terr, mil, res_civ, res_city)
        L.append("")
    L.append(strip_dead_relations(dipl_section(rec), rec))     # ★ 放在文明段之后；死亡玩家不出现
    return "\n".join(L).rstrip()


def strip_dead_relations(section: str, rec: dict) -> str:
    """Drop destroyed players from the relations matrix (their rows and their entries in other rows).
    The savegame keeps a War/Peace row for a dead civ forever; the review run 2026-09-15 saw six names
    in a four-civ world and enemy counts that could not be reconciled with the glossary."""
    dead = {p["name"] for p in rec["state"]["players"] if not p.get("is_alive")}
    if not dead:
        return section
    out = []
    for line in section.split("\n"):
        m = re.match(r"^- (.+?) -> (.*)$", line)
        if not m:
            out.append(line); continue
        if m.group(1) in dead:
            continue
        ents = [x for x in m.group(2).split("; ") if x.rsplit(": ", 1)[0] not in dead]
        out.append(f"- {m.group(1)} -> " + ("; ".join(ents) if ents else "no relations"))
    return "\n".join(out)


_WORLD_LEVEL = ("war_declared", "ceasefire_agreed", "ceasefire_expired", "peace_treaty_signed", "peace_established",
                "alliance_formed", "alliance_ended", "first_contact", "diplstate_change", "city_founded", "city_destroyed",
                "city_transferred", "government_change", "disaster_occurred", "tech_discovered")


def build_H(recs, frames, viewer, view: int, evmap, ruleset: str = "",
            a_prod: bool = True, a_milonly: bool = True, world: dict | None = None) -> tuple[str, dict]:
    """预热段：K 份报告 + K−1 段 INTERVAL；地图只在首帧。

    INTERVAL = bench.interval_v2.interval_block（2026-09-10 用户定稿）：五回合一段的新闻体，
    外交 → 城市 → 政体 → 灾害 → 科技 → 各文明新建单位 → 建筑/奇迹 → 地块工程 → 军事移动
    （按目的地聚合）→ 战斗（HP 按最大值百分比）。视角 2 的过滤在 interval_block 内部。
    `evmap` 必须是**完整**事件流（含 unit_built / action_finished_* / unit_lost），不能只给
    MAJOR_EVENT_TYPES。"""
    from bench.interval_v2 import interval_block          # 惰性：interval_v2 反过来 import 本模块的 visible_to
    L, stats = [], {"n_events": 0, "n_events_hidden": 0}
    rs_base = base_ruleset(recs[frames[0]].get("ruleset") or "civ2civ3")
    mil = unit_is_military(rs_base)
    if ruleset:
        L += ["===== RULES OF THIS WORLD  (static; stated once, like the map) =====",
              ruleset_primer(ruleset), ""]
    for i, t in enumerate(frames):
        rec = recs[t]
        if view == 1:
            rep = stat_report(rec, ruleset, with_map=(i == 0))
        else:
            rep = render_stale_report(rec, viewer, with_map=(i == 0), ruleset=ruleset)
        L.append(f"===== OBSERVATION FRAME {i+1}/{len(frames)}   turn {t}   "
                 + ("(with map)" if i == 0 else "(no map; terrain is static)") + " =====")
        if i == 0:
            # review run 2026-09-15: every reader asked what the two numeric city columns are
            L.append("(Legend: in the CITIES table the `food` and `shields` columns are the city's current "
                     "stockpiles -- the granary and the production box -- not per-turn output. "
                     "`founded` is the turn the city was founded.)")
        L.append(rep)
        if i + 1 < len(frames):
            a, b = t, frames[i + 1]
            # ★ 2026-09-15 (review run): an event recorded at turn t happens AFTER the turn-t snapshot
            # (measured: a city founded at record t exists in state t+1 in 34/35 cases, a discovered tech
            # in 102/111).  So the interval between frames a and b is [a, b), not (a, b].
            evs = [e for e in evmap if a <= (e.get("engine_turn") or e["turn"]) < b]
            if view == 2:
                hidden = sum(1 for e in evs if e.get("type") in _WORLD_LEVEL and not visible_to(e, viewer, rec))
                stats["n_events_hidden"] += hidden
            stats["n_events"] += len(evs)
            L.append("")
            L.append(f"===== INTERVAL   turn {a}..{b-1}   "
                     + ("(all civilizations)" if view == 1
                        else f"(only what {viewer} can know)") + " =====")
            L.append(interval_block(recs, a, b, evmap, mil, view, viewer, rs_base, world))
            L.append("")
    return "\n".join(L), stats


def build_A(evmap, t1, t2, viewer, view, rec, recs=None, ruleset="",
            milonly=True) -> str:
    """设计 1 的外生动作 A —— **与 H 里的 INTERVAL 段完全同格式**。

    同格式是刻意的：A 和 INTERVAL 描述的是同一种东西（一个区间里发生了什么），
    只是 INTERVAL 在历史段、A 在预测段。让它们长得一样，模型不需要学两套读法。

    ⚠️ A 覆盖的窗口只是 `t1..t1+Δ`，而预测的 horizon 可以远到 +300 —— 所以
    **落在 A 窗口内的 horizon 上，设计 1 接近 MDP；超出之后就不是**。
    读数必须按「horizon 是否在 A 窗口内」分层，否则两种性质不同的题被平均掉。
    单文明视角下 A 只含该文明能知道的（与 INTERVAL 用同一套知情过滤）。
    """
    evs = [e for e in evmap if t1 < (e.get("engine_turn") or e["turn"]) <= t2]
    if view == 2:
        evs = [e for e in evs if visible_to(e, viewer, rec)]
    L = ["Events:"] + (group_events(evs) or ["  (none)"])
    if recs is not None:
        for label, rows in interval_narrative(
                recs, t1, t2, unit_is_military(ruleset or "civ2civ3"),
                military_only=milonly):
            L.append(f"{label}:")
            L += rows
    return "\n".join(L)
