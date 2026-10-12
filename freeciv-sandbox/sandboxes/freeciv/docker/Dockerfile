# Server-only freeciv build for headless autogame data collection.
#
# --enable-debug is REQUIRED: the `timeout -1` autogame test mode (which lets a
# server run with no clients attached and no turn gating) is only compiled in on
# debug builds.  Without it the server waits forever for a client to connect.
#
# v3 (2026-09-19): adds patches/v3-ai-task-persist.patch (see the layer below).
# v2 (2026-09-09): applies patches/v2-engine.patch and ships modified rulesets.
# Everything the patch and the ruleset edits change is documented in
# bundle/docs/freeciv/09_benchmark_v2修改记录.md §8.1-8.3 and
# bench/measure/task2_trait_source_audit.md:
#   * Builder trait: three `0.5 + trait/50/2` integer divisions were a no-op for
#     the whole 30-90 range -> cast to double.
#   * Trader trait: `want /= (130*50/trait)` integer division -> float.
#   * First contact defaults to Peace (3.1 has no ruleset knob for this).
#   * Rulesets: trait ranges 10-150; civ2civ3 national trade routes enabled
#     (75% / 150%); two illness variants (civ2civ3_ill0, civ2civ3_ill30).
FROM debian:bookworm-slim AS build

ARG FREECIV_VERSION=3.1.6
ARG TASK_PERSIST=1
ARG http_proxy
ARG https_proxy

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential autoconf automake libtool pkg-config gettext \
      libcurl4-openssl-dev zlib1g-dev libbz2-dev liblzma-dev libzstd-dev \
      libicu-dev libreadline-dev libsqlite3-dev \
      curl ca-certificates xz-utils python3 patch \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
# Use the frozen upstream source; it has no pre-generated configure script.
RUN curl -fL --retry 3 https://github.com/freeciv/freeciv/archive/refs/tags/R3_1_6.tar.gz -o /src/freeciv.tar.gz \
 && echo "acf6d8027ddec4682287d2cbd38cc1479dbad6fe7676f3f6e7213c6bc2beb640  /src/freeciv.tar.gz" | sha256sum --check -
RUN tar xzf freeciv.tar.gz && mv freeciv-R3_1_6 freeciv

# ---- v2 engine patch: its own layer, before configure/make ----
COPY patches/v2-engine.patch /src/patches/v2-engine.patch
WORKDIR /src/freeciv
RUN patch -p1 --forward < /src/patches/v2-engine.patch \
 && grep -q "return DS_PEACE;" server/plrhand.c \
 && test "$(grep -c '(double) ai_trait_get_value(TRAIT_BUILDER' ai/default/daidomestic.c ai/default/daimilitary.c | awk -F: '{s+=$2} END{print s}')" = "3" \
 && grep -q "130.0 \* TRAIT_DEFAULT_VALUE" ai/default/daidomestic.c \
 && test "$(grep -c 'TRAIT_DEFAULT_VALUE \* 0.65' ai/default/daidomestic.c ai/default/daimilitary.c | awk -F: '{s+=$2} END{print s}')" = "3"

# ---- v2 diplomacy patch (added after pilot P1 2026-09-09): war must not be an
# absorbing state.  Ceasefire agreeable down to love -600 (was -400); a holding
# ceasefire rebuilds love fast enough to reach the peace clause inside its
# 16-turn window; the Aggressive trait's unprovoked-war threshold spans
# -360..-15 instead of -268..-178.
COPY patches/v2-diplomacy.patch /src/patches/v2-diplomacy.patch
RUN patch -p1 --forward < /src/patches/v2-diplomacy.patch \
 && grep -q "MAX_AI_LOVE \* 6 / 10" ai/default/daidiplomacy.c \
 && grep -q "aggr_sr / max_sr \* 1.9" ai/default/daidiplomacy.c \
 && grep -q "type == DS_CEASEFIRE" ai/default/daidiplomacy.c \
 && grep -q "MAX_AI_LOVE / 40" ai/default/daidiplomacy.c

# ---- v2 bloc patch (adopted 2026-09-09 after pilots p12d/e): stock AI made war
# an absorbing state via alliance blocs (95-98% of war pair-turns were "ally's
# enemy" wars; 84-86% of ceasefires were broken within 15 turns to help an ally).
# (1) a signed ceasefire is honoured, (2) peace -- not alliance -- may be signed
# with someone allied to an enemy, (3) peace is proposed during a ceasefire even
# if the partner fights an ally.  Measured: War 50% -> 33%, Cease-fire 3% -> 20%.
COPY patches/v2-bloc.patch /src/patches/v2-bloc.patch
RUN patch -p1 --forward < /src/patches/v2-bloc.patch \
 && test "$(grep -c 'wm-v2 (bloc)' ai/default/daidiplomacy.c)" = "2"

# ---- v2 aggressive patch (adopted 2026-09-09 after pilot p12f): the ceasefire
# love threshold and the post-declaration war stubbornness scale with the
# Aggressive trait.  These two are what make the trait identifiable
# (within-world Spearman -0.03 -> +0.42); the attacker-want multiplier and the
# help-ally draw that were also tested had no measurable effect and are not used.
COPY patches/v2-aggressive.patch /src/patches/v2-aggressive.patch
RUN patch -p1 --forward < /src/patches/v2-aggressive.patch \
 && test "$(grep -c 'wm-v2 (aggr' ai/default/daidiplomacy.c)" = "6"

# ---- v2 builder patch (adopted 2026-09-09 after pilot p12g): the military
# advisor's want is discounted by the Builder trait right where it competes with
# the domestic advisor (x1.67 at 10 .. x0.5 at 150).  Scaling building wants
# alone (v2-engine.patch) never moved the build mix; this does (within-world
# Spearman +0.01 -> +0.36).
COPY patches/v2-builder.patch /src/patches/v2-builder.patch
RUN patch -p1 --forward < /src/patches/v2-builder.patch \
 && grep -q "wm-v2 (builder)" ai/default/daicity.c

# ---- v3 AI-task persistence patch (2026-09-19, agent-play per-turn stepping): the
# default AI saved a unit's ferry/bodyguard links but not its current task, so a
# driver that reloads the game every turn had AI settlers that re-chose a site each
# turn, walked one tile and never founded: 71 foundings over 16 games x 100 turns vs
# 187 on the continuous mainline (doc 17 §6.7.3).  The task is now saved and restored
# (a build-city mission only when its goto target survived the load).
COPY patches/v3-ai-task-persist.patch /src/patches/v3-ai-task-persist.patch
RUN if [ "$TASK_PERSIST" = "1" ]; then \
      patch -p1 --forward < /src/patches/v3-ai-task-persist.patch \
      && test "$(grep -c 'wm-v3' ai/default/aiunit.c ai/default/aiunit.h ai/default/daisettler.c | awk -F: '{s+=$2} END{print s}')" = "5"; \
    else test "$TASK_PERSIST" = "0"; fi

RUN ./autogen.sh --no-configure-run \
 && ./configure \
      --prefix=/opt/freeciv \
      --enable-client=no \
      --enable-fcmp=no \
      --enable-debug=yes \
      --disable-nls \
      --enable-shared=no \
 && make -j"$(nproc)" \
 && make install

# ---- v2 rulesets: edit the installed copies, then derive the illness variants ----
WORKDIR /opt/freeciv/share/freeciv
RUN set -e; \
    for rs in civ2civ3 classic; do \
      sed -i -E 's/^(expansionist|trader|aggressive|builder)_min = 30$/\1_min = 10/; s/^(expansionist|trader|aggressive|builder)_max = 90$/\1_max = 150/' $rs/nations.ruleset; \
      test "$(grep -c '_min = 10$' $rs/nations.ruleset)" = "4"; \
      test "$(grep -c '_max = 150$' $rs/nations.ruleset)" = "4"; \
    done; \
    sed -i 's/^    "National",   0,     "Cancel",     "None"$/    "National",   75,    "Cancel",     "None"/; s/^    "NationalIC", 0,     "Cancel",     "None"$/    "NationalIC", 150,   "Cancel",     "None"/' civ2civ3/game.ruleset; \
    grep -q '"National",   75,' civ2civ3/game.ruleset; \
    grep -q '"NationalIC", 150,' civ2civ3/game.ruleset; \
    for lvl in 0 30; do \
      cp -r civ2civ3 civ2civ3_ill$lvl; cp civ2civ3.serv civ2civ3_ill$lvl.serv; \
      sed -i "s/^illness_base_factor = 10$/illness_base_factor = $lvl/" civ2civ3_ill$lvl/game.ruleset; \
      grep -q "^illness_base_factor = $lvl$" civ2civ3_ill$lvl/game.ruleset; \
    done


FROM debian:bookworm-slim
ARG http_proxy
ARG https_proxy
RUN apt-get update && apt-get install -y --no-install-recommends \
      libcurl4 zlib1g libbz2-1.0 liblzma5 libzstd1 libicu72 libreadline8 \
      libsqlite3-0 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /opt/freeciv /opt/freeciv
ENV PATH=/opt/freeciv/bin:$PATH
ENV FREECIV_DATA_PATH=/opt/freeciv/share/freeciv
WORKDIR /work
ENTRYPOINT []
CMD ["freeciv-server", "--version"]
