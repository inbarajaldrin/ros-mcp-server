"""Discovery 004: post-contact AUTO_success vs AUTO_fail divergence (u_orange).

Aligns each attempt on its own contact_idx_active (already physically derived as
the first |fz_t|_smooth > 5N sample by script 01). Resamples per-attempt features
onto a common [-0.5s, +5.0s] @ 100Hz grid, computes per-class p25/p50/p75
trajectories, and finds the first post-contact time at which AUTO_fail p50 leaves
the AUTO_success [p25, p75] band — for each feature.

Outputs metrics.json with per-feature t_div_s + per-timepoint band data for
follow-up analysis (no script change needed for plotting).

Usage:
    python3 13_post_contact_divergence_auto.py [out_path.json]
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import DATA_DIR, PER_SAMPLE_DIR  # noqa: E402

SUMMARIES = json.load(open(DATA_DIR / "summaries.json"))

T_PRE = -0.5
T_POST = 5.0
DT = 0.01
T_GRID = np.arange(T_PRE, T_POST + DT / 2, DT)
N_T = len(T_GRID)

FEATURES = [
    "fz_t", "F_lat_base", "F_lat_tool",
    "dz_dt_mm_s", "tilt_deg", "z_drop_mm",
    "xy_excursion_mm", "r_cop_m", "commanded_fz",
]

OBJECT = "u_orange"
SESSION_TAG = "20260504"
SUCCESS_THRESH_MM = 20.0
FAIL_THRESH_MM = 5.0


def classify(s):
    name = Path(s["csv_path"]).name
    if s["object"] != OBJECT or SESSION_TAG not in name:
        return None
    fdz = s.get("final_z_drop_mm") or 0.0
    if fdz >= SUCCESS_THRESH_MM:
        return "AUTO_success"
    if fdz < FAIL_THRESH_MM:
        return "AUTO_fail"
    return None  # exclude partials


def resample(t_src, x_src, t_dst):
    t_src = np.asarray(t_src, dtype=float)
    x_src = np.asarray(x_src, dtype=float)
    mask = np.isfinite(t_src) & np.isfinite(x_src)
    t_src = t_src[mask]
    x_src = x_src[mask]
    if len(t_src) < 2:
        return np.full_like(t_dst, np.nan, dtype=float)
    out = np.interp(t_dst, t_src, x_src, left=np.nan, right=np.nan)
    out[(t_dst < t_src[0]) | (t_dst > t_src[-1])] = np.nan
    return out


def smooth_pm(x, half_samples=5):
    """Centered ±half_samples mean smoother (NaN-aware)."""
    n = len(x)
    out = np.full(n, np.nan)
    for i in range(n):
        a = max(0, i - half_samples)
        b = min(n, i + half_samples + 1)
        w = x[a:b]
        m = np.isfinite(w)
        if m.any():
            out[i] = np.nanmean(w)
    return out


def main():
    trajs = {"AUTO_success": {f: [] for f in FEATURES},
             "AUTO_fail": {f: [] for f in FEATURES}}
    used = {"AUTO_success": [], "AUTO_fail": []}
    skipped = {"no_per_sample": 0, "no_contact_idx": 0, "wrong_class": 0}

    for r in SUMMARIES:
        cls = classify(r)
        if cls is None:
            skipped["wrong_class"] += 1
            continue
        ps_path = PER_SAMPLE_DIR / (Path(r["csv_path"]).stem + ".per_sample.json")
        if not ps_path.exists():
            skipped["no_per_sample"] += 1
            continue
        ps = json.load(open(ps_path))
        ci = r.get("contact_idx_active")
        if ci is None:
            skipped["no_contact_idx"] += 1
            continue
        t_active = np.asarray(ps["t_s"], dtype=float)
        if ci < 0 or ci >= len(t_active):
            skipped["no_contact_idx"] += 1
            continue
        t_rel = t_active - t_active[ci]
        used[cls].append(Path(r["csv_path"]).name)
        for f in FEATURES:
            arr = ps.get(f)
            if arr is None:
                trajs[cls][f].append(np.full(N_T, np.nan))
                continue
            x_aligned = resample(t_rel, arr, T_GRID)
            x_aligned = smooth_pm(x_aligned, half_samples=5)  # 100ms smoothing
            trajs[cls][f].append(x_aligned)

    n_succ = len(used["AUTO_success"])
    n_fail = len(used["AUTO_fail"])
    print(f"AUTO_success n={n_succ}  AUTO_fail n={n_fail}  skipped={skipped}")
    if n_succ < 5 or n_fail < 5:
        print("ERROR: insufficient samples")
        sys.exit(1)

    out = {
        "object": OBJECT,
        "session_tag": SESSION_TAG,
        "n_AUTO_success": n_succ,
        "n_AUTO_fail": n_fail,
        "t_grid_s": T_GRID.tolist(),
        "thresholds": {"success_mm": SUCCESS_THRESH_MM, "fail_mm": FAIL_THRESH_MM},
        "smoothing_pm_ms": 50,
        "features": {},
        "ranking": [],
    }

    for f in FEATURES:
        succ_stack = np.stack(trajs["AUTO_success"][f])
        fail_stack = np.stack(trajs["AUTO_fail"][f])
        with np.errstate(all="ignore"):
            succ_p25 = np.nanpercentile(succ_stack, 25, axis=0)
            succ_p50 = np.nanpercentile(succ_stack, 50, axis=0)
            succ_p75 = np.nanpercentile(succ_stack, 75, axis=0)
            fail_p25 = np.nanpercentile(fail_stack, 25, axis=0)
            fail_p50 = np.nanpercentile(fail_stack, 50, axis=0)
            fail_p75 = np.nanpercentile(fail_stack, 75, axis=0)

        # First-divergence time: first t > 0 where fail_p50 leaves [succ_p25, succ_p75]
        post = np.where(T_GRID >= 0.0)[0]
        t_div_s = None
        gold_at_div = succ_at_div = fail_at_div = None
        gap_at_div = None
        sustain_samples_required = 10  # 100 ms sustained divergence
        run = 0
        for i in post:
            if not (np.isfinite(fail_p50[i]) and np.isfinite(succ_p25[i]) and np.isfinite(succ_p75[i])):
                run = 0
                continue
            outside = fail_p50[i] < succ_p25[i] or fail_p50[i] > succ_p75[i]
            if outside:
                run += 1
                if run >= sustain_samples_required:
                    j = i - sustain_samples_required + 1
                    t_div_s = float(T_GRID[j])
                    succ_at_div = float(succ_p50[j])
                    fail_at_div = float(fail_p50[j])
                    gap_at_div = float(fail_at_div - succ_at_div)
                    break
            else:
                run = 0

        out["features"][f] = {
            "n_succ": int(np.isfinite(succ_stack).any(axis=0).sum()),
            "n_fail": int(np.isfinite(fail_stack).any(axis=0).sum()),
            "succ_p25": [None if not np.isfinite(v) else float(v) for v in succ_p25],
            "succ_p50": [None if not np.isfinite(v) else float(v) for v in succ_p50],
            "succ_p75": [None if not np.isfinite(v) else float(v) for v in succ_p75],
            "fail_p25": [None if not np.isfinite(v) else float(v) for v in fail_p25],
            "fail_p50": [None if not np.isfinite(v) else float(v) for v in fail_p50],
            "fail_p75": [None if not np.isfinite(v) else float(v) for v in fail_p75],
            "t_div_s": t_div_s,
            "succ_p50_at_div": succ_at_div,
            "fail_p50_at_div": fail_at_div,
            "gap_at_div": gap_at_div,
        }

    # Ranking: features sorted by earliest sustained divergence
    ranked = []
    for f, d in out["features"].items():
        if d["t_div_s"] is None:
            continue
        ranked.append({
            "feature": f, "t_div_s": d["t_div_s"],
            "succ_p50": d["succ_p50_at_div"], "fail_p50": d["fail_p50_at_div"],
            "gap": d["gap_at_div"],
        })
    ranked.sort(key=lambda x: x["t_div_s"])
    out["ranking"] = ranked

    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("metrics.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")

    print("\nRanked feature divergences (sustained 100ms outside AUTO_success [p25,p75] band):")
    print(f"{'feature':18s}  t_div(s)   succ_p50    fail_p50    gap")
    for r in ranked:
        print(f"  {r['feature']:18s}  {r['t_div_s']:>+5.2f}    "
              f"{r['succ_p50']:>+8.3f}   {r['fail_p50']:>+8.3f}   {r['gap']:>+8.3f}")
    if not ranked:
        print("  (no feature diverges)")


if __name__ == "__main__":
    main()
