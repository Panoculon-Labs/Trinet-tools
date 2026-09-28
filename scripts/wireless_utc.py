#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Panoculon Labs. Part of the Trinet toolkit.
"""Put memory-card recordings on UTC using the phone's wireless status log.

    python scripts/wireless_utc.py phone.jsonl.gz --recordings /media/CARD1 /media/CARD2
    python scripts/wireless_utc.py a.jsonl b.jsonl --recordings cards/ -o out/ --per-frame
    python scripts/wireless_utc.py phone.jsonl.gz --recordings cards/ --write-sidecars
    python scripts/wireless_utc.py phone.jsonl.gz --inspect

Keep the Trinet phone app monitoring while the cameras record, export its
wireless log, then run this. Every take on the cards gets the UTC of its first
and last frame (and optionally of every frame) with a 1-sigma uncertainty and a
confidence. Writes wireless_utc.json and wireless_utc.csv (one row per take and
camera eye). See docs/wireless_utc.md for the method and how to read the output.

Exit status: 0 = done (every take resolved, or --strict not given);
2 = --strict and some take is unmatched / ambiguous / below --min-confidence;
1 = error (unreadable log, no recordings found, bad arguments).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trinet_tools import wireless_utc as wu  # noqa: E402
from trinet_tools.wireless_log import load_logs  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+", metavar="LOG", help="wireless log export(s): .jsonl, .jsonl.gz or legacy .json")
    ap.add_argument("--recordings", nargs="+", metavar="DIR",
                    help="card mount points or folders to search (recursively) for takes")
    ap.add_argument("-o", "--outdir", default=".", help="output directory (default: current)")
    ap.add_argument("--json", default="wireless_utc.json", help="JSON report name (in OUTDIR)")
    ap.add_argument("--csv", default="wireless_utc.csv", help="CSV report name (in OUTDIR)")
    ap.add_argument("--per-frame", action="store_true",
                    help="also write <take>_<eye>.utc.csv with the UTC of every frame")
    ap.add_argument("--write-sidecars", action="store_true",
                    help="write <take>.utc.json next to each resolved take")
    ap.add_argument("--force", action="store_true", help="overwrite existing .utc.json sidecars")
    ap.add_argument("--unit", help="unit id (8 hex) to assume for takes that carry none")
    ap.add_argument("--fit", choices=["local", "global", "live"], default="local",
                    help="clock refit: local (default), one global line, or the phone's live fit")
    ap.add_argument("--window-s", type=float, default=600.0, help="local fit half-window, s (default 600)")
    ap.add_argument("--max-extrapolation-s", type=float, default=1800.0,
                    help="how far outside a logged run a take may lie, s (default 1800)")
    ap.add_argument("--phone-utc", choices=["best", "system", "network", "sntp", "corrected"],
                    default="best", help="phone time source (default best: SNTP > network > system)")
    ap.add_argument("--latency-correction-ms", type=float, default=0.0,
                    help="subtract this radio latency from every UTC (default 0; bias is 0..1.5 ms)")
    ap.add_argument("--pick", action="append", default=[], metavar="TAKEPATH=STORE:SEGMENT",
                    help="resolve an ambiguous take by naming the log segment it belongs to")
    ap.add_argument("--min-confidence", choices=["low", "medium", "high"], default="low",
                    help="results below this are reported as below_min_confidence")
    ap.add_argument("--strict", action="store_true", help="exit 2 if any take is not resolved")
    ap.add_argument("--inspect", action="store_true", help="describe the log(s) and exit")
    args = ap.parse_args(argv)

    for p in args.logs:
        if not Path(p).is_file():
            print(f"error: no such log file: {p}", file=sys.stderr)
            return 1
    try:
        log = load_logs(args.logs)
    except (OSError, ValueError) as e:
        print(f"error: cannot read log: {e}", file=sys.stderr)
        return 1

    picks = {}
    for spec in args.pick:
        if "=" not in spec:
            print(f"error: --pick expects TAKEPATH=STORE:SEGMENT, got {spec!r}", file=sys.stderr)
            return 1
        k, v = spec.split("=", 1)
        picks[k] = v
    cfg = wu.MatchConfig(fit=args.fit, window_s=args.window_s,
                         max_extrapolation_s=args.max_extrapolation_s, phone_utc=args.phone_utc,
                         latency_correction_ms=args.latency_correction_ms,
                         min_confidence=args.min_confidence, picks=picks, unit=args.unit)

    if args.inspect:
        print("\n".join(wu.inspect_log(log, cfg)))
        return 0
    if not args.recordings:
        print("error: --recordings DIR is required (or use --inspect)", file=sys.stderr)
        return 1

    for w in log.warnings:
        print(f"warning: {w}", file=sys.stderr)
    files = wu.discover_takes(args.recordings)
    if not files:
        print("error: no recordings found under " + ", ".join(args.recordings), file=sys.stderr)
        return 1
    takes = [wu.load_take(f, args.unit) for f in files]
    runs = wu.build_runs(log)
    print(f"log: {len(log.files)} file(s), {len(log.units)} camera(s), {len(runs)} clock run(s); "
          f"recordings: {len(takes)} take(s)")

    matches = wu.match_takes(takes, runs, log, cfg)
    results = [wu.take_to_utc(m, per_frame=args.per_frame) for m in matches]
    kits = wu.kit_consistency(results)
    params = {k: v for k, v in vars(args).items() if k not in ("logs", "inspect")}
    prov = wu.provenance(log, params)

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    wu.write_json(out / args.json, results, kits, prov)
    wu.write_csv(out / args.csv, results)
    print("\n".join(wu.summarize(results, kits)))
    print(f"\nwrote {out / args.json}\nwrote {out / args.csv}")
    if args.per_frame:
        n = sum(len(wu.write_per_frame(out, t)) for t in results)
        print(f"wrote {n} per-frame CSV file(s) to {out}")
    if args.write_sidecars:
        for t in results:
            if t.match.status != "resolved":
                continue
            p, written = wu.write_sidecar(t, prov, force=args.force)
            print(("wrote " if written else "kept existing (use --force to overwrite) ") + str(p))

    unresolved = sum(1 for t in results if t.match.status != "resolved")
    if unresolved and args.strict:
        print(f"{unresolved} take(s) not resolved (--strict)", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
