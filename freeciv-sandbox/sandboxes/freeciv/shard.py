"""Compressed-file helpers shared by the collector and the branch runner.

v2 keeps every savegame (mainline per turn, arm per 10 turns): ~1-2 TB of plain
text, which zstd brings down ~7x and reads back 4-5x faster than gzip.  The
corpus *shards* (`world.jsonl.gz`, `branch.jsonl.gz`, `events.jsonl.gz`) stay
gzip on purpose: every reader in `bundle/` and `bench/` already speaks gzip and
their volume (~10 GB) does not justify touching all of them.  So: zstd for the
new savegame archives only.

`zstandard` is preferred; without it we shell out to /usr/bin/zstd.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

try:
    import zstandard as _zstd
except Exception:            # pragma: no cover
    _zstd = None

LEVEL = 9


def compress_file(src: Path, dst: Path, *, remove_src: bool = False) -> Path:
    """Write `dst` (.zst) from `src`.  Atomic via a .part rename."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    if _zstd is not None:
        c = _zstd.ZstdCompressor(level=LEVEL)
        with open(src, "rb") as f, open(tmp, "wb") as g:
            c.copy_stream(f, g)
    else:
        with open(tmp, "wb") as g:
            subprocess.run(["zstd", f"-{LEVEL}", "-q", "-c", str(src)],
                           stdout=g, check=True)
    tmp.replace(dst)
    if remove_src:
        Path(src).unlink(missing_ok=True)
    return dst


def read_text(path: Path, encoding: str = "utf-8") -> str:
    """Read a .zst (or plain) text file."""
    path = Path(path)
    if path.suffix != ".zst":
        return path.read_text(encoding=encoding)
    if _zstd is not None:
        with open(path, "rb") as f:
            return _zstd.ZstdDecompressor().stream_reader(f).read().decode(encoding)
    return subprocess.run(["zstd", "-d", "-q", "-c", str(path)],
                          capture_output=True, check=True).stdout.decode(encoding)


def decompress_file(src: Path, dst: Path) -> Path:
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if _zstd is not None:
        with open(src, "rb") as f, open(dst, "wb") as g:
            _zstd.ZstdDecompressor().copy_stream(f, g)
    else:
        with open(dst, "wb") as g:
            subprocess.run(["zstd", "-d", "-q", "-c", str(src)], stdout=g, check=True)
    return dst
