# Ruleset Primer

两个 ruleset 的静态规则前言，供放在 H 开头（和地图一样，静态的东西只给一次）。

| 文件 | tokens | 覆盖 |
|---|---|---|
| `RULES_civ2civ3.md` | ≈ 4,190 | train / valid / test-iid 的全部世界 |
| `RULES_classic.md` | ≈ 4,333 | **test-ood 的 100 个 XL 世界** |

⚠️ **两个都要用。** OOD 那 100 个世界是 classic ruleset，只给 civ2civ3 的话，
"答错是因为不会推演还是不知道规则"这个混杂在 OOD 上原封不动地存在 —— 而 OOD
恰恰是最需要干净归因的地方。

## 内容与来源

| 段 | 条目 | 来源 |
|---|---|---|
| 科技树（前置） | 87 | `techs.ruleset` 的 `req1`/`req2` |
| 单位 | 56 | `units.ruleset`：class / 科技需求 / 费用 / 攻防 / HP / 火力 / 移动 / 被谁淘汰 |
| 建筑 | 73 | `buildings.ruleset`：genus / 科技前置 / 费用 / 维护 |
| 地形 | 13 | `terrain.ruleset`：移动消耗 / 防御加成 / 粮盾贸易 / 开矿灌溉增益 |
| 政体 | 9 | 见下 |

地形是 13 而非 14：`Inaccessible` 被过滤掉了（不可进入的占位地形）。

## ★ 政体是刻意区别对待的

政体的数值**不在 `governments.ruleset` 里**，而是散在 `effects.ruleset` 的需求向量
DSL 中 —— civ2civ3 共 492 条效果，其中 113 条以政体为条件，带 present/absent 取反，
还有注释掉的行。**机器总结它就是"一份解读"，而 preamble 里一条自信的错规则比没有规则更糟。**

所以：

- **civ2civ3** → 直接抄 `README.civ2civ3` 里**作者自己写的数值表**（税率 / 单位维护 /
  自由单位 / 战争不满 / 军管 / 内战率 / 帝国规模 / 腐败与浪费 / 按距离衰减 +
  每个政体的 Special），一字未改。
- **classic** 没有这张表 → 用作者的 `helptext` 原文，加上**忠实转录**所有以政体为条件
  的效果（形如 `Output_Waste = 37 when Gov=Despotism; OutputType=Trade`），不做任何改写。

## 重新生成

```bash
# ruleset 文件在 docker 镜像里，不在语料里
docker run --rm --entrypoint sh -v /tmp/rs:/out wm/freeciv-server:3.1.6 -c \
  'for r in civ2civ3 classic; do mkdir -p /out/$r; \
     cp /opt/freeciv/share/freeciv/$r/*.ruleset /opt/freeciv/share/freeciv/$r/README.* /out/$r/; done; \
   chmod -R a+r /out'
python3 -m sandboxes.freeciv.ruleset --root /tmp/rs/civ2civ3 --out RULES_civ2civ3.md
```

## ⚠️ 解析器不能复用 savegame.py

存档写 `key=value`（无空格），ruleset 写 `name\t= _("X")` —— 实测 `savegame.py` 的
`_KV_RE` 对 ruleset 每一行都不匹配。`ruleset.py` 是独立模块，处理了：任意 `[section]`
头、引号外的 `;` 注释、527 处反斜杠续行、`_()` gettext 包裹、`?unitclass:` 限定前缀、
以及多行 `{...}` 需求表（**建筑的科技前置就在里面** —— 只读标量会让所有建筑都显示"无前置"）。

primer 末尾会打印 `(counts: 87 technologies, 56 units, 73 improvements, 13 terrains)`，
与原文条目数交叉核对，防止解析漏段。
