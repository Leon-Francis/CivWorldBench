"""Benchmark 的报告渲染：统计式，而不是逐单位清单。

为什么改：`UNITS` 段占 tier-S 报告 **25.8%** 的 token、tier-M **29.5%**，而逐单位的
`hp/moves/坐标` 在 Δ=100 的跨度上基本不可预测 —— 实测「~70% 的可赢空间是逐城逐单位簿记」，
所以纯 next-token loss 主要在奖励名册更新，而不是世界因果。把单位压成**计数**直接砍掉
其中最不可预测的那一半，同时留下「有多少兵、多少工人」这个真正有信息的部分。

分工变成：
  报告  = 状态的**聚合**（有什么、多少）
  事件  = **发生了什么**（谁对谁做了什么）
这比原来的划分更贴 Dreamer（观测是状态、动作是转移的叙述），而且消掉了报告与 actions
之间实测的重合（税率/生产/单位位置三项 100% 重复）。

⚠️ 刻意住在 benchmark 侧、不进 `report.py`：报告格式是 **benchmark 的身份**（要冻结进
manifest），而 `report.py` 的 `full`/`compact` 要跟着语料演进。两者混在一处，任何一边动
都会污染另一边。`_fmt_event` 的情况相反 —— 事件词表必须跟着语料，所以那个复用 `report.py`。

⚠️ 一处已知的信息损失：去掉单位坐标就看不到「边境上有军队集结」。设计把它移到事件叙事里
（「某单位在某处攻击」），但那只在**已经开打**之后才出现，集结阶段是盲的。v1 连攻击叙事
都没有（引擎事件流要等 v2），所以 v1 的报告对军事集结完全不可见 —— 写进已知边界。
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@lru_cache(maxsize=4)
def unit_is_military(ruleset: str) -> dict[str, bool]:
    """单位类型 -> 是否军事，从 ruleset primer 的 UNITS 段读 `attack` 列。

    判据是 `attack > 0`。用 primer 而不是硬编码列表，因为两个 ruleset 的单位表不同
    （OOD 是 classic），硬编码会在 OOD 上静默错分。
    """
    ruleset = re.sub(r"_ill\d+$", "", ruleset)   # v2: rulesetdir variants share the base rules
    f = ROOT / f"bundle/docs/freeciv/ruleset_primer/RULES_{ruleset}.md"
    out: dict[str, bool] = {}
    if not f.exists():
        return out
    sec = re.search(r"^## UNITS\n(.*?)(?=^## )", f.read_text(encoding="utf-8"), re.S | re.M)
    if not sec:
        return out
    for line in sec.group(1).splitlines():
        parts = [x.strip() for x in line.split("|")]
        if len(parts) < 6 or parts[0].startswith("unit"):
            continue
        try:
            out[parts[0]] = int(parts[4]) > 0        # attack
        except ValueError:
            continue
    return out


def unit_counts(units: list[dict], mil: dict[str, bool]) -> str:
    """`units (12): Workers x4, Settlers x3, ... | military 4, civilian 8`"""
    import collections
    c = collections.Counter(u["type"] for u in units)
    body = ", ".join(f"{k} x{v}" for k, v in c.most_common())
    m = sum(v for k, v in c.items() if mil.get(k, False))
    unknown = sum(v for k, v in c.items() if k not in mil)
    tail = f"military {m}, civilian {len(units) - m - unknown}"
    if unknown:
        tail += f", unclassified {unknown}"          # primer 里没有的单位类型，不猜
    return f"units ({len(units)}): {body or '(none)'}\n    {tail}"


CITY_HDR = ("city | x,y | size | founded | building | food | shields | buildings(';') "
            "| worked-tile resources")


def city_rows(cities: list[dict], owner: str = "", res_city=None) -> list[str]:
    """城市表。★ 去掉 `owner` 列（嵌在文明块下是冗余的），加该城工作地块的资源统计。"""
    out = []
    for c in sorted(cities, key=lambda z: (z["y"], z["x"])):
        b = c.get("improvements") or {}
        names = sorted(k for k, v in b.items() if v) if isinstance(b, dict) else []
        row = (f"    {c['name']} | {c['x']},{c['y']} | {c['size']} | {c['founded']} | "
               f"{c['production']} | {c['food_stock']} | {c['shield_stock']} | "
               + ("; ".join(names) if names else "none"))
        if res_city is not None:
            row += " | " + _fmt_counter(res_city.get((owner, c["name"])))
        out.append(row)
    return out


def civ_block(p: dict, terr: dict, mil: dict[str, bool],
              res_civ=None, res_city=None) -> list[str]:
    """一个文明的完整统计块 —— CIVILIZATIONS + CITIES + UNITS 三段合一。"""
    t = terr.get(p["name"]) or {}
    # ⚠️ trade / production / science_output / tax_pct 都在嵌套的 `income` 里，
    # 不是 player 的顶层字段（顶层的 `trade`/`production`/`science` 恒为 null）。
    inc = p.get("income") or {}
    L = [f"{p['name']} of the {p['nation']}",
         f"  government: {p['government']}        treasury: {p['gold']} gold",
         f"  trade {inc.get('trade', 0)}, production {inc.get('production', 0)}, "
         f"science output {inc.get('science_output', 0)}",
         f"  revenue allocation: {inc.get('tax_pct', 0)}% treasury, "
         f"{inc.get('science_pct', 0)}% research, {inc.get('luxury_pct', 0)}% luxury"]
    L += [f"  population: {p['population']} thousand     score {p['score']}, "
          f"culture {p['culture']}",
          f"  technologies ({len(p['techs'])}): " + ", ".join(sorted(p["techs"])),
          # v2 (09 §3.2c): what is being researched right now and how far along.
          # `research` (the long-term goal) is what v1 printed; `research_now` is the
          # v2 field; bulbs come via bench.render.annotate_research.  No cost (leaks sciencebox).
          # ⚠️ state.Player.bulbs_last is misnamed: it is research.bulbs, the ACCUMULATED
          # points toward the current tech (verified turn by turn); per-turn income is
          # income.science_output.  No cost: cost would print the hidden sciencebox.
          f"  research: {('Future Tech' if p.get('research_now') == 'A_FUTURE' else p.get('research_now')) or 'nothing'}"
          + (f" [{p.get('bulbs_last', 0)}/{p['_cost']}]" if p.get("_cost") else f" [{p.get('bulbs_last', 0)} accumulated]")
          + (f"    goal: {p['research']}" if p.get("research") and p["research"] not in ("A_UNSET", "A_NONE", "None") else ""),
          f"  small wonders: " + (", ".join(p.get("small_wonders") or []) or "none"),
          f"  territory: {t.get('tiles','?')} tiles "
          f"({t.get('land_tiles','?')} land, {t.get('worked_tiles','?')} worked)",
          f"  resources in my territory: "
          + (_fmt_counter(res_civ.get(p["name"])) if res_civ else "n/a"),
          f"  cities ({len(p['cities'])}):", "    " + CITY_HDR]
    L += city_rows(p["cities"], p["name"], res_city)
    L.append("  " + unit_counts(p["units"], mil))
    return L


# --------------------------------------------------------------------------
# 区间叙事：从状态 diff 派生的细节动作
#
# ⚠️ v1 没有引擎事件流，只有状态差可用。哪些能派生、哪些不能，全部是实测的：
#
#   ✅ 城市生产了某单位  新单位落在自己城市格上，实测 341/445 = 76.6% 可归因。
#                      未归因的多是 Trireme 出现在海面（运输/登船把位置变化误判成新
#                      单位）—— 那些**不猜**，单独列成 "appeared"。
#   ✅ 单位移动          同 id 追踪起点→终点，精确。20 回合内中期 51% 的单位动过；
#                      只报军事单位省 60% 行数。
#   ✅ 单位损失          **事实照写**：某单位在某处不再存在。
#   ✅ 城市陷落          `city_transferred`（易手，带 from_player）与 `city_destroyed`
#                      （size-1 城被攻下即毁）**都已经是事件**，所以「攻陷了某城」这个
#                      事实在 v1 就有。
#   ❌ **攻击方是谁**    只有这一项做不到。231 个消失单位里仅 82 个旁边有敌军，其余
#                      149 个原因不明（饥荒 / 解散 / 被运走）。用「附近有敌军」去推断
#                      会把那 149 里的一部分误报成战斗 —— 与旧 events.py 把每次征服
#                      报成「毁城+建城」是同一类错误（那批数据 84% 的 city_destroyed
#                      是假的）。⇒ 只陈述损失与陷落，不写「谁打的」。
#                      v2 的引擎流会直接给 destroyer / unit_lost.reason / action_*。
# --------------------------------------------------------------------------

def _index(rec):
    u = {v["id"]: (p["name"], v) for p in rec["state"]["players"] for v in p["units"]}
    c = {(x["x"], x["y"]): (p["name"], x["name"])
         for p in rec["state"]["players"] for x in p["cities"]}
    return u, c


def interval_narrative(recs: dict, a: int, b: int, mil: dict[str, bool],
                       military_only: bool = True, cap: int = 30) -> list[tuple[str, list]]:
    """`turn a+1 .. b` 内的细节动作，返回 [(段名, 行列表)]，由调用方与 Events 并列排版。

    ★ **逐回合追踪，不比两端。** 比两端会犯两个错，都实测过：

      ① 归因错。一个单位在 t=60 于某城出生、到 t=76 已走开 —— 在 t=76 检查它的位置
         当然不在城里，于是被报成「来源不明」。实测 turn 57..76：比两端只能归因 47%，
         逐回合追踪 67%。这与旧 `events.py` 按每玩家分别 diff city-id、把易手报成
         「毁城+建城」是**同一类错误**（那批数据 84% 的 city_destroyed 是假的）。
      ② 移动漏报。只报「两端都存在」的单位就排除了区间内新建的那批 —— 而它们恰好是
         刚出城赶路的。实测同一区间：两端都在仅 35 个、区间内新建 43 个，于是军事移动
         只剩 1 条。

    ⇒ 需要区间内**每一回合**的记录，不只两个端点。

    排版按「区间聚合」而非逐回合：生产按 (文明, 城) 归并、移动按 (文明, 单位类型) 归并、
    损失按文明归并。回合号不写 —— 区间本身就是时间粒度。
    ⚠️ 「来源不明的出现」整段去掉：剩下 33% 归因不上的多是 Trireme 在城里造好当回合就
    下水，写出来只是噪声；`Units produced` 少报一条比报一条假的好。
    """
    import collections
    turns = [t for t in range(a, b + 1) if t in recs]
    if len(turns) < 2:
        return []
    idx = {t: _index(recs[t]) for t in turns}
    X = recs[turns[-1]]["geo"]["xsize"]

    first_seen, last_seen = {}, {}
    for t in turns:
        for uid, (own, u) in idx[t][0].items():
            first_seen.setdefault(uid, (t, own, u))
            last_seen[uid] = (t, own, u)
    at_start, at_end = set(idx[turns[0]][0]), set(idx[turns[-1]][0])

    # -- 生产：按 (文明, 城, 坐标) 归并，值是 类型 -> 个数 -------------------
    prod = collections.defaultdict(collections.Counter)
    for uid, (t0, own, u) in first_seen.items():
        if uid in at_start:
            continue
        cf = idx[t0][1]                                # 首次出现那一回合的城市表
        key = (u["x"], u["y"])
        if key in cf and cf[key][0] == own:
            prod[(own, cf[key][1], key)][u["type"]] += 1
    by_civ = collections.defaultdict(list)
    for (own, city, key), c in prod.items():
        by_civ[own].append(f"{city}({key[0]},{key[1]}) "
                           + ", ".join(f"{k} x{v}" for k, v in c.most_common()))
    produced = [f"  {own}: " + "; ".join(sorted(v)) for own, v in sorted(by_civ.items())]

    # -- 移动：按 (文明, 类型) 归并，一行多个单位 ---------------------------
    mv = collections.defaultdict(list)
    for uid, (t0, own, u0) in first_seen.items():
        _, _, u1 = last_seen[uid]
        if (u0["x"], u0["y"]) == (u1["x"], u1["y"]):
            continue
        if military_only and not mil.get(u1["type"], False):
            continue
        mv[(own, u1["type"])].append(f"#{uid} {u0['x']},{u0['y']}->{u1['x']},{u1['y']}")
    moved = [f"  {own} {typ}: " + "; ".join(v)
             for (own, typ), v in sorted(mv.items())]

    # -- 损失：按文明归并 -------------------------------------------------
    ls = collections.defaultdict(list)
    for uid, (t1, own, u) in last_seen.items():
        if uid in at_end:
            continue
        ls[own].append(f"{u['type']} #{uid} at {u['x']},{u['y']}")
    lost = [f"  {own}: " + "; ".join(v) for own, v in sorted(ls.items())]

    def trim(rows):
        return rows[:cap] + ([f"  ... and {len(rows)-cap} more"] if len(rows) > cap else [])

    out = []
    for label, rows in (("Units produced", produced),
                        ("Military movement" if military_only else "Unit movement", moved),
                        ("Units lost", lost)):
        if rows:
            out.append((label, trim(rows)))
    return out


# --------------------------------------------------------------------------
# 资源统计：替代逐地块的行式清单
#
# 原来的 tile detail 是一行一地图行的清单（`y=0  11:i  12:i  23:n`），可读性差且长。
# ⚠️ 曾经试过把它改成「每层一张网格」，实测**更大** —— extras 是稀疏的
# （3,840 格里只有 1,250 格有内容），网格要为每个空格也花一个字符。
# ⇒ 改成**聚合统计**：每个文明疆域内多少、每座城工作的地块里多少。
# 全局那一行去掉了 —— 逐文明的已经够用。
# 地形网格（`### THE MAP` 的第一部分）保持原样，它是密的、网格形式本来就合适。
# --------------------------------------------------------------------------

RES_CLASS = {
    "food":    ("Wheat", "Fruit", "Oasis", "Fish", "Whales", "Buffalo", "Pheasant", "Ivory"),
    "shield":  ("Coal", "Iron", "Resources", "Peat", "Oil"),
    "trade":   ("Gold", "Silk", "Wine", "Gems", "Spice", "Furs", "Dyes"),
    "terrain": ("River",),
}
_INFRA = ("Road", "Railroad", "Irrigation", "Farmland", "Mine")


def _res_class(name: str) -> str:
    for k, v in RES_CLASS.items():
        if name in v:
            return k
    return "other"


def resource_stats(rec: dict) -> tuple[dict, dict, dict]:
    """(全局, 逐文明疆域内, 逐城工作地块内) 的资源与基建计数。

    ⚠️ `owner` / `worked` 是**逗号分隔的字符串**，不是逐字符网格 —— 按字符遍历会取到
    `-` 和 `,` 本身（第一版就是这样，文明名渲染成了 "-" 和 ","）。
    `owner` 的值是玩家**槽位号**（0/1/2，`-` = 无主），`worked` 是**城市 id**。
    """
    import collections
    g = rec["geo"]["grids"]
    ex = g.get("extras") or []
    own = [row.split(",") if isinstance(row, str) else (row or [])
           for row in (g.get("owner") or [])]
    wk = [row.split(",") if isinstance(row, str) else (row or [])
          for row in (g.get("worked") or [])]
    slot2name = {str(i): p["name"] for i, p in enumerate(rec["state"]["players"])}
    world = collections.Counter()
    per_civ = collections.defaultdict(collections.Counter)
    per_city = collections.defaultdict(collections.Counter)
    id2city = {str(c["id"]): (p["name"], c["name"])
               for p in rec["state"]["players"] for c in p["cities"]}
    for y, row in enumerate(ex):
        for x, names in enumerate(row or []):
            for nm in (names or []):
                key = f"infra:{nm}" if nm in _INFRA else f"{_res_class(nm)}:{nm}"
                world[key] += 1
                o = own[y][x] if y < len(own) and x < len(own[y]) else "-"
                if o in slot2name:
                    per_civ[slot2name[o]][key] += 1
                w = wk[y][x] if y < len(wk) and x < len(wk[y]) else "-"
                if w in id2city:
                    per_city[id2city[w]][key] += 1
    return world, per_civ, per_city


def _fmt_counter(c, by_class: bool = True) -> str:
    """`food 7 (Wheat 3, Fish 4) · shield 5 (Coal 5) · infra 12 (Road 9, Mine 3)`"""
    import collections
    if not c:
        return "none"
    if not by_class:
        return ", ".join(f"{k.split(':')[-1]} {v}" for k, v in c.most_common())
    grp = collections.defaultdict(collections.Counter)
    for k, v in c.items():
        cls, nm = k.split(":", 1)
        grp[cls][nm] += v
    parts = []
    for cls in ("food", "shield", "trade", "terrain", "other", "infra"):
        if cls not in grp:
            continue
        tot = sum(grp[cls].values())
        inner = ", ".join(f"{n} {v}" for n, v in grp[cls].most_common())
        parts.append(f"{cls} {tot} ({inner})")
    return " · ".join(parts)
