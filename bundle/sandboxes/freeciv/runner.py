"""The one place that starts a freeciv server container.

This code existed in three copies -- `collect.py:_run_server`,
`collect.py:_branch_worker` and `verify_ops.py:run_turns` -- and the copies had
drifted: different subprocess timeouts (1800 / 900 / 600), different
`saveturns` (5 / 5 / 1), one of them silently ignoring a non-zero exit code.
Drift like that turns "the branch is short" into a data bug rather than an
error, so there is now a single implementation and the differences are
parameters.

Two facts about this server build that the settings below encode:

  * `timeout -1` puts the server in autogame test mode, which **only exists in
    a debug build** -- hence `--enable-debug=yes` in the Dockerfile.
  * `ec_turns 0` stops the event cache from being written into the savegame.
    Without it the saves carry wall-clock-flavoured state and the determinism
    check in `scripts/freeciv_determinism_check.sh` fails.

Lua interventions are injected as `lua cmd <one line>` entries placed *after*
the settings and *before* `start`, which is how the operators verified in
`docs/freeciv/探针状态与反事实算子设计.md` §2.5.6 were run.  ⚠️ `lua cmd`
accepts a single line only and **discards syntax errors without any output**,
so a malformed line looks exactly like a missing API.  Never conclude an
operator worked from the log; read the resulting state back.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

DOCKER_IMG = "wm/freeciv-server:3.1.6"

# Settings every run needs.  `victories ""` and `endspaceship disabled` keep a
# world from terminating early on a win condition, which would truncate a
# trajectory for reasons unrelated to the horizon we asked for.
BASE_SETTINGS: dict[str, Any] = {
    "timeout": -1,
    "minplayers": 0,
    "ec_turns": 0,
    "compresstype": "PLAIN",
    "autosaves": '"TURN|GAMEOVER"',
    "victories": '""',
    "endspaceship": "disabled",
}


@dataclass
class RunResult:
    ok: bool
    saves: list[Path] = field(default_factory=list)
    final: Path | None = None
    turn_reached: int = -1
    log: str = ""
    returncode: int = -1
    message: str = ""


def serv_script(settings: dict[str, Any], lua: Iterable[str] = (),
                commands: Iterable[str] = ()) -> str:
    """Build a .serv script.  Order is load-bearing: settings, then lua, then start."""
    lines = [f"set {k} {v}" for k, v in settings.items()]
    lines += [c for c in commands]
    for line in lua:
        if "\n" in line:
            raise ValueError("lua cmd takes a single line; multi-line input is "
                             "silently discarded by the server")
        lines.append(f"lua cmd {line}")
    lines.append("start")
    return "\n".join(lines) + "\n"


def run(workdir: Path, settings: dict[str, Any], *,
        start_sav: Path | None = None,
        lua: Iterable[str] = (),
        commands: Iterable[str] = (),
        timeout: int = 3600,
        image: str = DOCKER_IMG) -> RunResult:
    """Run one server to completion.

    `start_sav` None means a fresh game (the settings must then describe the map
    and players); otherwise the save is loaded and continued.  Returns the saves
    the run produced, sorted, plus the turn actually reached -- callers need that
    to tell "ran to the horizon" from "died early", which a zero exit code alone
    does not tell you.
    """
    from .savegame import Savegame

    # ⚠️ Absolute, always.  `docker run -v <path>:/work` reads a relative path as
    # a *named volume*, not a host directory, and refuses it with
    # "includes invalid characters for a local volume name" and rc=125 -- one
    # error per world, for every world, looking like a simulator failure.  Every
    # existing caller happened to pass an absolute path, so this only surfaced
    # when a new one did not.
    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    saves_dir = workdir / "saves"
    saves_dir.mkdir(exist_ok=True)

    cmd = ["docker", "run", "--rm", "--user", f"{os.getuid()}:{os.getgid()}",
           "-v", f"{workdir}:/work", image, "freeciv-server"]
    if start_sav is not None:
        if Path(start_sav) != workdir / "start.sav":
            shutil.copy(start_sav, workdir / "start.sav")
        cmd += ["-f", "/work/start.sav"]
    (workdir / "run.serv").write_text(serv_script(settings, lua, commands))
    cmd += ["-r", "/work/run.serv", "-e", "-s", "/work/saves", "-l", "/work/game.log"]

    try:
        # freeciv writes its log and console output in Latin-1 (the savegames
        # themselves are UTF-8).  This used to decode as UTF-8 with
        # errors="replace", which never raised but silently replaced every
        # non-ASCII byte: `Sükhbaatar` came out as `S�khbaatar`, so any civ or
        # city name in the log was corrupted.  Latin-1 cannot fail and round-trips
        # every byte.
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              encoding="latin-1", timeout=timeout)
        rc, stderr = proc.returncode, proc.stderr
    except subprocess.TimeoutExpired:
        rc, stderr = -9, f"timed out after {timeout}s"

    log = ""
    logp = workdir / "game.log"
    if logp.exists():
        log = logp.read_text(encoding="latin-1")

    saves = sorted(saves_dir.glob("*.sav"))
    if not saves:
        return RunResult(ok=False, log=log, returncode=rc,
                         message=f"no save produced; rc={rc}; "
                                 f"tail={(log or stderr)[-400:]}")
    final = saves[-1]
    reached = Savegame.load(final).turn
    return RunResult(ok=True, saves=saves, final=final, turn_reached=reached,
                     log=log, returncode=rc, message=f"reached turn {reached}")


def run_to_turn(workdir: Path, start_sav: Path, end_turn: int, *,
                save_every: int = 5, lua: Iterable[str] = (),
                extra: dict[str, Any] | None = None,
                timeout: int = 3600) -> RunResult:
    """Continue a save until `end_turn`.  Flags a short run as not-ok.

    A run that stops early is the failure mode that matters here: it looks like
    a valid but shorter branch, and averaging it in with full-length branches
    quietly biases every effect size computed from that horizon.
    """
    settings = dict(BASE_SETTINGS)
    settings.update({"saveturns": save_every, "endturn": end_turn})
    if extra:
        settings.update(extra)
    r = run(workdir, settings, start_sav=start_sav, lua=lua, timeout=timeout)
    if r.ok and r.turn_reached < end_turn - save_every:
        r.ok = False
        r.message = f"short run: reached {r.turn_reached}, wanted {end_turn}"
    return r


def cleanup(workdir: Path, keep: Iterable[Path] = ()) -> int:
    """Delete a run's savegames once they are parsed.  Returns bytes freed.

    Branch arms are the bulk of the collection and each one leaves ~40 MB of
    text saves behind; with 150 slots that is the difference between 6 GB of
    scratch and filling the disk.
    """
    keep_r = {Path(k).resolve() for k in keep}
    freed = 0
    d = Path(workdir) / "saves"
    if not d.exists():
        return 0
    for p in d.glob("*.sav"):
        if p.resolve() in keep_r:
            continue
        freed += p.stat().st_size
        p.unlink()
    return freed
