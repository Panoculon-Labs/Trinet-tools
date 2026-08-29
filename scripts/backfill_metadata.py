#!/usr/bin/env python3
"""Backfill metadata fields into already-delivered per-clip ZIPs.

For batches that shipped before a required field existed (e.g. the delivery
spec's ``environment_type``), this rewrites the ``metadata.json`` inside each
ZIP to add the field -- without re-collecting or re-uploading the recordings.
Everything else in the ZIP is copied through byte-for-byte.

    python3 scripts/backfill_metadata.py DELIVERIES/ \
        --environment residential/laundry

    # different environments per file? drive it from a CSV of
    #   zip_name,environment           (or episode,environment)
    python3 scripts/backfill_metadata.py DELIVERIES/ --map env_by_clip.csv

Standard-library Python 3 only. By default it rewrites ZIPs in place (via a
temp file + atomic replace); use --out DIR to write copies instead.
"""

import argparse
import csv
import io
import json
import os
import sys
import tempfile
import zipfile


def parse_environment(value):
    if "/" not in value:
        raise argparse.ArgumentTypeError("use TYPE/SUBCATEGORY, e.g. "
                                         "residential/laundry")
    t, s = value.split("/", 1)
    t = t.strip().lower()
    s = s.strip().lower().replace(" ", "_").replace("-", "_")
    if not t or not s:
        raise argparse.ArgumentTypeError("both TYPE and SUBCATEGORY required")
    return t, s


def load_map(path):
    """CSV of (key, environment) where key matches the zip name or clip id."""
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) < 2 or not row[0].strip():
                continue
            key = row[0].strip()
            if key.lower() in ("zip", "zip_name", "episode", "clip", "file"):
                continue                              # header
            out[key] = parse_environment(row[1].strip())
    return out


def _apply(meta, env, country, session, note):
    """Inject the fields into a metadata dict. Returns list of changes."""
    changed = []
    if env:
        etype, sub = env
        if meta.get("environment_type") != etype:
            meta["environment_type"] = etype
            changed.append("environment_type")
        if meta.get("environment_subcategory") != sub:
            meta["environment_subcategory"] = sub
            changed.append("environment_subcategory")
        e = meta.get("environment")
        if not isinstance(e, dict):
            e = {}
        e["type"], e["subcategory"] = etype, sub
        if note:
            e["note"] = note
        meta["environment"] = e
    if country:
        loc = meta.get("location")
        if not isinstance(loc, dict):
            loc = {}
        if loc.get("country") != country:
            loc["country"] = country
            changed.append("location.country")
        meta["location"] = loc
    if session and meta.get("session_id") != session:
        meta["session_id"] = session
        changed.append("session_id")
    return changed


def env_for(zip_name, clip_id, default_env, env_map):
    if env_map:
        base = zip_name[:-4] if zip_name.lower().endswith(".zip") else zip_name
        for key in (zip_name, base, clip_id):
            if key and key in env_map:
                return env_map[key]
        return None                                   # no mapping -> skip
    return default_env


def backfill_zip(path, out_path, default_env, env_map, country, session,
                 note, dry_run):
    """Rewrite one ZIP's metadata.json. Returns (status, detail)."""
    try:
        with zipfile.ZipFile(path, "r") as z:
            names = z.namelist()
            if "metadata.json" not in names:
                return "skip", "no metadata.json"
            meta = json.loads(z.read("metadata.json").decode("utf-8"))
            clip_id = meta.get("clip_id", "")
            env = env_for(os.path.basename(path), clip_id, default_env, env_map)
            if env is None:
                return "skip", "no environment mapping for this file"
            changes = _apply(meta, env, country, session, note)
            if not changes:
                return "unchanged", "already present"
            if dry_run:
                return "would-fix", ", ".join(changes)
            payload = {n: z.read(n) for n in names if n != "metadata.json"}
            infos = {i.filename: i for i in z.infolist()}
    except (OSError, zipfile.BadZipFile, ValueError, KeyError) as e:
        return "error", str(e)

    new_meta = (json.dumps(meta, indent=2) + "\n").encode("utf-8")
    tmp = out_path + ".part"
    try:
        with zipfile.ZipFile(tmp, "w", allowZip64=True) as z:
            for n in names:
                if n == "metadata.json":
                    z.writestr("metadata.json", new_meta, zipfile.ZIP_DEFLATED)
                else:
                    ci = infos[n]
                    z.writestr(ci, payload[n], compress_type=ci.compress_type)
        os.replace(tmp, out_path)
    except OSError as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        return "error", str(e)
    return "fixed", ", ".join(changes)


def build_parser():
    p = argparse.ArgumentParser(
        prog="backfill_metadata.py",
        description="Add missing metadata (e.g. environment_type) to delivered "
                    "per-clip ZIPs, in place.")
    p.add_argument("path", help="A delivery ZIP, or a folder of them.")
    p.add_argument("--environment", type=parse_environment, metavar="TYPE/SUB",
                   help="Environment to write into every file, "
                        "e.g. residential/laundry.")
    p.add_argument("--map", metavar="CSV",
                   help="CSV of zip_name-or-clip,environment for per-file "
                        "environments (overrides --environment).")
    p.add_argument("--country", metavar="CC",
                   help="Also set location.country (e.g. IN).")
    p.add_argument("--session-id", metavar="ID",
                   help="Also set session_id (rarely wanted in bulk).")
    p.add_argument("--env-note", metavar="TEXT", help="Environment note.")
    p.add_argument("--out", metavar="DIR",
                   help="Write fixed copies here instead of rewriting in place.")
    p.add_argument("--recursive", "-r", action="store_true",
                   help="Recurse into sub-folders.")
    p.add_argument("--dry-run", action="store_true",
                   help="Report what would change; write nothing.")
    return p


def find_zips(root, recursive):
    if os.path.isfile(root):
        return [root] if root.lower().endswith(".zip") else []
    hits = []
    for dirpath, _dirs, files in os.walk(root):
        for n in sorted(files):
            if n.lower().endswith(".zip"):
                hits.append(os.path.join(dirpath, n))
        if not recursive:
            break
    return hits


def main(argv=None):
    args = build_parser().parse_args(argv)
    env_map = load_map(args.map) if args.map else None
    if not env_map and not args.environment and not args.country \
            and not args.session_id:
        print("error: nothing to backfill -- pass --environment, --map, "
              "--country or --session-id.")
        return 2

    zips = find_zips(args.path, args.recursive)
    if not zips:
        print("error: no .zip files found at %s" % args.path)
        return 2
    if args.out and not args.dry_run:
        os.makedirs(args.out, exist_ok=True)

    tally = {}
    for zp in zips:
        out_path = (os.path.join(args.out, os.path.basename(zp))
                    if args.out and not args.dry_run else zp)
        status, detail = backfill_zip(
            zp, out_path, args.environment, env_map, args.country,
            args.session_id, args.env_note, args.dry_run)
        tally[status] = tally.get(status, 0) + 1
        mark = {"fixed": "[fix ]", "would-fix": "[dry ]",
                "unchanged": "[ ok ]", "skip": "[skip]",
                "error": "[FAIL]"}.get(status, "[????]")
        print("%s %s%s" % (mark, os.path.basename(zp),
                           "  -- " + detail if detail else ""))

    print("\n" + ", ".join("%d %s" % (c, s) for s, c in sorted(tally.items())))
    return 1 if tally.get("error") else 0


if __name__ == "__main__":
    sys.exit(main())
