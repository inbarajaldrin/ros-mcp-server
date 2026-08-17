#!/usr/bin/env python3
"""Derive the SEARCH press-force duty cycle from operator GOLD demos.

Motivating observation (2026-08-16): an autonomous u_brown SEARCH crawled at
1.24 mm/s against a commanded 5.0 mm/s and never explored past 2.70 mm of a
configured 8 mm spiral radius. Achieved |F_lat| (median 1.36 N) sat right on the
breakaway friction implied by the press force (|fz| median 4.31 N), so the peg
was stiction-pinned. Skill section 12 states the mechanism directly: "lateral
push without intermittent Fz release means peg slides across rim without falling
into chamfer."

This script measures HOW MUCH the operator unloads during the drag, so the
autonomous SEARCH can reproduce it.

Design constraint — the metric must generalize across all four FMB1 parts
without per-part tuning. So the headline figure is a SELF-NORMALIZING duty
cycle: the fraction of the search window spent below a fraction of that
episode's own median press. An absolute newton threshold cannot transfer
between a single-peg part and a multi-prong one (skill section 19.8); a ratio can.

Per skill section 15, features are computed on the tool-frame wrench
(wrench_frame_id == tool0_controller) and never on world-frame values.
Per skill section 4, the window is recomputed from raw CSV columns; no FSM
stdout label or `outcome` flag is trusted. The only external label used is
`hole_observed_operator.t_s` — the operator's SIGUSR1 timestamp, which is
ground truth by construction.

Usage:
    python3 30_press_duty_cycle.py [--logs-dir DIR] [--k 0.5]
"""

import argparse
import csv
import glob
import json
import os
import re
import statistics as st
from datetime import datetime

NAME_RE = re.compile(r"insert_(.+)_(\d{8}_\d{6})\.meta\.json$")

# Contact onset: |fz| above this, sustained, recomputed from raw. Used only to
# find the START of the drag window -- not as a control threshold.
CONTACT_N = 3.0
CONTACT_SUSTAIN_S = 0.10


def episode_start_epoch(meta):
    iso = meta.get("start_iso")
    if not iso:
        return None
    try:
        return datetime.fromisoformat(iso).timestamp()
    except ValueError:
        return None


def load_rows(csv_path):
    with open(csv_path) as fh:
        return list(csv.DictReader(fh))


def contact_onset_t(rows):
    """First t_s where |fz| stays above CONTACT_N for CONTACT_SUSTAIN_S."""
    run_start = None
    for r in rows:
        try:
            t = float(r["t_s"])
            fz = abs(float(r["fz"]))
        except (KeyError, ValueError):
            continue
        if fz > CONTACT_N:
            if run_start is None:
                run_start = t
            elif t - run_start >= CONTACT_SUSTAIN_S:
                return run_start
        else:
            run_start = None
    return None


def window_stats(rows, t0, t1, k):
    """Press statistics over [t0, t1] in tool frame."""
    fz = []
    times = []
    for r in rows:
        try:
            t = float(r["t_s"])
        except (KeyError, ValueError):
            continue
        if t0 is not None and t < t0:
            continue
        if t1 is not None and t > t1:
            continue
        try:
            fz.append(abs(float(r["fz"])))
            times.append(t)
        except (KeyError, ValueError):
            continue
    if len(fz) < 20:
        return None
    med = st.median(fz)
    if med <= 0:
        return None
    below_abs = sum(1 for v in fz if v < CONTACT_N) / len(fz)
    below_rel = sum(1 for v in fz if v < k * med) / len(fz)
    return {
        "n": len(fz),
        "dur_s": times[-1] - times[0],
        "median_fz": med,
        "p05_fz": sorted(fz)[len(fz) // 20],
        "duty_abs": below_abs,   # fraction under CONTACT_N  (per-part sensitive)
        "duty_rel": below_rel,   # fraction under k*median   (self-normalizing)
    }


def classify(meta):
    """GOLD = operator marked the hole. Otherwise autonomous, split by seat."""
    if meta.get("hole_observed_operator"):
        return "GOLD"
    return "AUTO_OK" if str(meta.get("outcome")) == "success" else "AUTO_FAIL"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs-dir",
                    default=os.path.expanduser(
                        "~/Documents/ros-mcp-server/compliant_insertion_studio/logs"))
    ap.add_argument("--k", type=float, default=0.5,
                    help="relative-unload threshold as a fraction of episode median |fz|")
    args = ap.parse_args()

    buckets = {}
    skipped = 0
    for meta_path in sorted(glob.glob(os.path.join(args.logs_dir, "insert_*.meta.json"))):
        name_match = NAME_RE.search(os.path.basename(meta_path))
        if not name_match:
            continue
        obj = name_match.group(1)
        try:
            meta = json.load(open(meta_path))
        except (json.JSONDecodeError, OSError):
            skipped += 1
            continue

        csv_path = meta_path.replace(".meta.json", ".csv")
        if not os.path.exists(csv_path):
            skipped += 1
            continue
        try:
            rows = load_rows(csv_path)
        except OSError:
            skipped += 1
            continue
        if not rows or rows[0].get("wrench_frame_id") != "tool0_controller":
            skipped += 1   # skill section 15: tool frame only
            continue

        kind = classify(meta)
        t_contact = contact_onset_t(rows)
        if t_contact is None:
            skipped += 1
            continue

        # GOLD closes the window at the operator's SIGUSR1 mark; autonomous runs
        # have no such label, so the window runs to the end of the episode.
        t_end = None
        if kind == "GOLD":
            start_epoch = episode_start_epoch(meta)
            mark = meta["hole_observed_operator"].get("t_s")
            if start_epoch and mark:
                t_end = mark - start_epoch
                if not (t_contact < t_end <= float(rows[-1]["t_s"])):
                    t_end = None   # mark outside the logged window; drop episode
            if t_end is None:
                skipped += 1
                continue

        stats = window_stats(rows, t_contact, t_end, args.k)
        if stats is None:
            skipped += 1
            continue
        buckets.setdefault((obj, kind), []).append(stats)

    print(f"skipped episodes (no contact / wrong frame / unreadable): {skipped}\n")
    print(f"Self-normalizing duty cycle = fraction of drag window with "
          f"|fz| < {args.k} x that episode's median |fz|\n")
    header = ("| object | kind | n | median |fz| (N) | duty_rel | duty_abs (<3N) |")
    print(header)
    print("|---|---|---|---|---|---|")
    for obj in sorted({o for o, _ in buckets}):
        for kind in ("GOLD", "AUTO_OK", "AUTO_FAIL"):
            eps = buckets.get((obj, kind))
            if not eps:
                continue
            med = st.median([e["median_fz"] for e in eps])
            drel = st.median([e["duty_rel"] for e in eps])
            dabs = st.median([e["duty_abs"] for e in eps])
            print(f"| {obj} | {kind} | {len(eps)} | {med:.2f} | "
                  f"{drel*100:.1f}% | {dabs*100:.1f}% |")

    # Cross-object spread is the generalization test: a metric that varies wildly
    # between parts cannot be a single tuned constant.
    print()
    for kind in ("GOLD", "AUTO_OK", "AUTO_FAIL"):
        vals = []
        for (obj, k2), eps in buckets.items():
            if k2 == kind and eps:
                vals.append(st.median([e["duty_rel"] for e in eps]))
        if len(vals) >= 2:
            print(f"{kind:10s} duty_rel across objects: "
                  f"min {min(vals)*100:.1f}%  max {max(vals)*100:.1f}%  "
                  f"spread {(max(vals)-min(vals))*100:.1f} pts")


if __name__ == "__main__":
    main()
