"""Verify and unpack the release data into the paths used by the benchmark code."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
MAX_ARCHIVE_BYTES = 95 * 1024**2
MAX_UNPACKED_BYTES = 90 * 1024**2


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def safe_path(name, roots=("bench", "paper", "data_v2")):
    if not isinstance(name, str) or any(ord(char) < 32 for char in name) or "\\" in name or ":" in name:
        raise ValueError("Invalid data path")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != name:
        raise ValueError("Unsafe data path: " + name)
    if len(path.parts) < 2 or path.parts[0] not in roots:
        raise ValueError("Unexpected data path: " + name)
    target = ROOT.joinpath(*path.parts)
    if any(parent.is_symlink() for parent in (target, *target.parents)):
        raise ValueError("Data destination contains a symlink: " + name)
    return target


def valid_digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def validate_archive(archive):
    source = safe_path(archive["path"], roots=("data",))
    if not source.name.endswith(".tar.gz"):
        raise ValueError("Unexpected archive format")
    if (not isinstance(archive["group"], str) or not archive["group"]
            or not valid_digest(archive["sha256"])):
        raise ValueError("Invalid archive index entry")
    for field, limit in (("bytes", MAX_ARCHIVE_BYTES), ("unpacked_bytes", MAX_UNPACKED_BYTES)):
        if type(archive[field]) is not int or not 0 <= archive[field] <= limit:
            raise ValueError("Invalid archive size")
    records = archive["files"]
    if not isinstance(records, list) or not records:
        raise ValueError("Archive must contain indexed files")
    for record in records:
        safe_path(record["path"])
        if (type(record["bytes"]) is not int or not 0 <= record["bytes"] <= MAX_UNPACKED_BYTES
                or not valid_digest(record["sha256"])):
            raise ValueError("Invalid member index entry")
    expected = {record["path"]: record for record in records}
    if len(expected) != len(records):
        raise ValueError("Duplicate paths in data index")
    if sum(record["bytes"] for record in records) != archive["unpacked_bytes"]:
        raise ValueError("Unpacked size differs from data index")
    return source, expected


def validate_index(index):
    archive_paths, member_paths = set(), set()
    for archive in index["archives"]:
        _, expected = validate_archive(archive)
        if archive["path"] in archive_paths or member_paths.intersection(expected):
            raise ValueError("Duplicate paths across data archives")
        archive_paths.add(archive["path"])
        member_paths.update(expected)
    for name in member_paths:
        if any(parent.as_posix() in member_paths for parent in PurePosixPath(name).parents):
            raise ValueError("A data path is both a file and a directory")


def existing_matches(target, record):
    if not target.exists():
        return False
    if (not target.is_file() or target.stat().st_size != record["bytes"]
            or sha256(target) != record["sha256"]):
        raise ValueError("Existing file differs; move it before unpacking: " + record["path"])
    return True


def validate_member(member, expected, seen):
    if not member.isfile() or member.name not in expected or member.name in seen:
        raise ValueError("Unexpected archive member: " + member.name)
    if (member.uid or member.gid or member.uname or member.gname or member.linkname
            or member.mtime != 0 or member.mode != 0o644 or member.sparse is not None
            or any(key != "path" or value != member.name for key, value in member.pax_headers.items())):
        raise ValueError("Unexpected archive metadata: " + member.name)
    if member.size != expected[member.name]["bytes"]:
        raise ValueError("Member size differs from data index: " + member.name)


def unpack(archive):
    source, expected = validate_archive(archive)
    if (not source.is_file() or source.stat().st_size != archive["bytes"]
            or sha256(source) != archive["sha256"]):
        raise ValueError("Archive checksum failed: " + archive["path"])
    for name, record in expected.items():
        existing_matches(safe_path(name), record)
    seen = set()
    written = 0
    # Verify the complete archive before publishing any restored files.
    with tempfile.TemporaryDirectory(prefix=".unpack-", dir=ROOT) as staging:
        pending = {}
        with tarfile.open(source, "r|gz") as stream:
            if stream.pax_headers:
                raise ValueError("Unexpected global archive metadata")
            for member in stream:
                validate_member(member, expected, seen)
                seen.add(member.name)
                record = expected[member.name]
                temporary = Path(staging) / str(len(seen))
                digest = hashlib.sha256()
                size = 0
                with stream.extractfile(member) as payload, temporary.open("wb") as output:
                    for chunk in iter(lambda: payload.read(1024 * 1024), b""):
                        size += len(chunk)
                        digest.update(chunk)
                        output.write(chunk)
                if size != record["bytes"] or digest.hexdigest() != record["sha256"]:
                    raise ValueError("Member checksum failed: " + member.name)
                temporary.chmod(0o644)
                pending[member.name] = temporary
        if seen != set(expected):
            raise ValueError("Archive is missing indexed data files")
        if sha256(source) != archive["sha256"]:
            raise ValueError("Archive changed during extraction")
        for name, temporary in pending.items():
            target = safe_path(name)
            if existing_matches(target, expected[name]):
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target = safe_path(name)
            try:
                os.link(temporary, target, follow_symlinks=False)
                written += 1
            except FileExistsError:
                if target.is_symlink():
                    raise ValueError("Data destination became a symlink: " + name)
                existing_matches(target, expected[name])
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List data groups without extracting")
    parser.add_argument("--groups", help="Extract only these comma-separated groups; default: all")
    args = parser.parse_args()
    if not (ROOT / "data/index.json").is_file():
        parser.error("Dataset: In preparation. This code release contains no benchmark data. No local archive index is available.")
    index = json.loads((ROOT / "data/index.json").read_text(encoding="utf-8"))
    validate_index(index)
    groups = {archive["group"] for archive in index["archives"]}
    selected = set(args.groups.split(",")) if args.groups else groups
    if selected - groups:
        parser.error("Unknown groups: " + ", ".join(sorted(selected - groups)))
    total = 0
    for archive in index["archives"]:
        if archive["group"] not in selected:
            continue
        if args.list:
            print(f"{archive['path']}: {len(archive['files'])} files, "
                  f"{archive['bytes'] / 1024**2:.1f} MiB download, "
                  f"{archive['unpacked_bytes'] / 1024**2:.1f} MiB unpacked")
        else:
            written = unpack(archive)
            total += written
            print(f"Verified {archive['path']}; extracted {written} files", flush=True)
    if not args.list:
        print(f"Data ready: {total} newly extracted files.")


if __name__ == "__main__":
    main()
