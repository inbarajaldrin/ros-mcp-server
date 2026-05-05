#!/usr/bin/env python3
# Reference: full analysis pipeline. Run after every batch of new telemetry.
# Idempotent: re-running over the same logs produces the same outputs.
# Ralph loop entry point: `python3 analysis/scripts/run_all.py`
import os, sys, subprocess, time
sys.path.insert(0, os.path.dirname(__file__))
from _paths import DATA_DIR, ITERATIONS_DIR, ANALYSIS_DIR

HERE = os.path.dirname(os.path.abspath(__file__))
STAGES = [
    ("01_extract.py",            "parse raw CSVs to per-sample features"),
    ("02_bin_by_depth.py",       "depth-banded medians + IQRs by group"),
    ("03_search_phase.py",       "pre-collapse window stats"),
    ("04_direction_vs_seat.py",  "F_lat direction vs contact->seat geometry"),
    ("05_fail_motion_pattern.py","success vs failure motion divergence"),
    ("06_discriminator.py",      "Fz-collapse classifier validation"),
]

def main():
    print(f"data dir: {DATA_DIR}")
    print(f"iterations dir: {ITERATIONS_DIR}")
    t0 = time.time()
    for script, desc in STAGES:
        print(f"\n=== {script} — {desc} ===")
        rc = subprocess.call([sys.executable, os.path.join(HERE, script)])
        if rc != 0:
            print(f"FAIL: {script} exited {rc}")
            sys.exit(rc)
    print(f"\nall stages OK in {time.time()-t0:.1f}s")
    print(f"outputs:")
    print(f"  {DATA_DIR}/summaries.json")
    print(f"  {DATA_DIR}/bin_stats.json")
    print(f"  {DATA_DIR}/search_phase.json")
    print(f"  {DATA_DIR}/per_sample/*.json")

if __name__ == "__main__":
    main()
