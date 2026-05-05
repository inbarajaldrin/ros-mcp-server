#!/usr/bin/env python3
# Reference: score a single ralph iteration's results vs the prior best.
# Reads summaries.json, computes the headline metric for the iteration's CSV subset,
# and writes RESULTS.md into the iteration directory. Returns nonzero if the metric regressed.
#
# Usage:
#   python3 score_iteration.py <iter_dir> [--csv-glob 'insert_u_orange_20260505_*.csv']
#
# The iteration directory must contain HYPOTHESIS.md describing what was tried.
# This script appends RESULTS.md based on the new telemetry.
import argparse, glob, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from _paths import DATA_DIR, LOG_DIR, ITERATIONS_DIR

HEADLINE = "durable_collapse_rate"  # primary metric the loop optimizes


def smooth(arr, w=50):
    n=len(arr); out=[None]*n; buf=[]
    for i,v in enumerate(arr):
        if v is not None and v==v:
            buf.append(v)
            if len(buf)>w: buf.pop(0)
            out[i]=sum(buf)/len(buf) if buf else None
    return out


def detect_durable_collapse(ps, ci, fs=100.0):
    fz_t = ps["fz_t"]; dz_dt = ps["dz_dt_mm_s"]
    fz_s = smooth(fz_t, 50); dz_s = smooth(dz_dt, 50)
    sustain_n = int(0.5 * fs)
    run = 0
    for i in range(ci, len(fz_s)):
        v = fz_s[i]; d = dz_s[i]
        if v is None or v != v: run = 0; continue
        if abs(v) < 2.0 and (d is not None and d == d) and d < -2.0:
            run += 1
            if run >= sustain_n: return True
        else: run = 0
    return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("iter_dir", help="iteration directory (e.g. iterations/001-baseline)")
    p.add_argument("--csv-glob", default=None, help="restrict to this CSV glob (default: all in summaries.json)")
    args = p.parse_args()

    iter_dir = os.path.abspath(args.iter_dir)
    if not os.path.isdir(iter_dir):
        print(f"FAIL: {iter_dir} not a dir"); sys.exit(2)

    summaries_path = os.path.join(str(DATA_DIR), "summaries.json")
    if not os.path.exists(summaries_path):
        print(f"FAIL: run_all.py first to produce {summaries_path}"); sys.exit(2)
    summaries = json.load(open(summaries_path))

    if args.csv_glob:
        keep = set(os.path.basename(p) for p in glob.glob(os.path.join(str(LOG_DIR), args.csv_glob)))
        summaries = [r for r in summaries if os.path.basename(r["csv_path"]) in keep]

    n = len(summaries)
    if n == 0:
        print("FAIL: zero episodes after filter"); sys.exit(2)

    n_seated = sum(1 for r in summaries if r["final_z_drop_mm"] >= 20.0)
    durable = 0
    for r in summaries:
        ps_path = os.path.join(str(DATA_DIR), "per_sample",
                               os.path.basename(r["csv_path"]).replace(".csv", ".per_sample.json"))
        if not os.path.exists(ps_path): continue
        ps = json.load(open(ps_path))
        if detect_durable_collapse(ps, r["contact_idx_active"], r["fs_hz"]):
            durable += 1

    metrics = {
        "n_episodes": n,
        "seated_rate": n_seated / n if n else 0.0,
        "durable_collapse_rate": durable / n if n else 0.0,
        "median_final_z_drop_mm": sorted(r["final_z_drop_mm"] for r in summaries)[n//2] if n else 0.0,
    }

    # Compare to prior iteration
    prev = None
    iters = sorted(os.path.basename(d) for d in glob.glob(os.path.join(str(ITERATIONS_DIR), "*"))
                   if os.path.isdir(d))
    cur_name = os.path.basename(iter_dir)
    for name in iters:
        if name >= cur_name: continue
        prev_metrics_path = os.path.join(str(ITERATIONS_DIR), name, "metrics.json")
        if os.path.exists(prev_metrics_path):
            prev = json.load(open(prev_metrics_path))

    delta_str = ""
    if prev:
        d = metrics[HEADLINE] - prev["metrics"][HEADLINE]
        delta_str = f" (delta vs prior: {d:+.3f})"

    # write metrics.json (machine-readable)
    out = {"iteration": cur_name, "headline": HEADLINE, "metrics": metrics, "prior": prev}
    json.dump(out, open(os.path.join(iter_dir, "metrics.json"), "w"), indent=2)

    # write RESULTS.md
    md = [f"# Results — {cur_name}\n"]
    md.append(f"**Headline:** `{HEADLINE}` = **{metrics[HEADLINE]:.3f}**{delta_str}\n")
    md.append(f"**Episodes scored:** {n}\n")
    md.append("\n| metric | value |")
    md.append("|---|---|")
    for k, v in metrics.items():
        md.append(f"| {k} | {v:.3f}" + (" |" if isinstance(v, float) else f" {v} |"))
    if prev:
        md.append(f"\n**Prior iteration:** {prev['iteration']}, headline = {prev['metrics'][HEADLINE]:.3f}")
    with open(os.path.join(iter_dir, "RESULTS.md"), "w") as fh:
        fh.write("\n".join(md))

    print(f"{cur_name}  {HEADLINE}={metrics[HEADLINE]:.3f}{delta_str}  (n={n})")
    if prev and metrics[HEADLINE] < prev["metrics"][HEADLINE]:
        print("REGRESSION — exit 1"); sys.exit(1)


if __name__ == "__main__":
    main()
