#!/usr/bin/env python3
# Reference: align every episode on the contact moment, crop to [-1s, +6s], compute per-timepoint
# success-manifold and find the first-divergence-time for each failure.
#
# This converts 132 "failures" into 132 negative-class probes of state-action space, and produces
# a per-feature "leaves the success band at t=X" signal for the discovery loop.
import json, os, sys, math
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from _paths import DATA_DIR

SUMMARIES = json.load(open(os.path.join(str(DATA_DIR), "summaries.json")))

# Reference window: 1 s pre-contact to 6 s post-contact (covers full operator demo + most failures pre-timeout).
# Sample at 100 Hz -> 700 timesteps.
T_PRE = -1.0
T_POST = 6.0
DT = 0.01
T_GRID = np.arange(T_PRE, T_POST + DT/2, DT)
N_T = len(T_GRID)

FEATURES = ["fz_t", "F_lat_base", "tcp_x", "tcp_y", "tcp_z", "dz_dt_mm_s", "tilt_deg", "commanded_fz", "z_drop_mm"]


def resample(t_src, x_src, t_dst):
    """Linear interp; returns NaN outside src range."""
    t_src = np.asarray(t_src, dtype=float)
    x_src = np.asarray(x_src, dtype=float)
    mask = np.isfinite(t_src) & np.isfinite(x_src)
    t_src = t_src[mask]; x_src = x_src[mask]
    if len(t_src) < 2:
        return np.full_like(t_dst, np.nan, dtype=float)
    out = np.interp(t_dst, t_src, x_src, left=np.nan, right=np.nan)
    out[(t_dst < t_src[0]) | (t_dst > t_src[-1])] = np.nan
    return out


def classify(r):
    """Return one of: GOLD, FAIL, AUTO_SEATED, OTHER."""
    obj = r["object"]; out = r["outcome"]; csv = r["csv_path"]
    if "20260503" in csv and out == "success":
        return "GOLD"
    if "20260504" in csv and out == "abort" and r["final_z_drop_mm"] < 5.0:
        return "FAIL"
    if "20260504" in csv and r["final_z_drop_mm"] >= 25.0:
        return "AUTO_SEATED"
    return "OTHER"


def main():
    # gather aligned trajectories
    trajs = {"GOLD": {}, "FAIL": {}, "AUTO_SEATED": {}}
    for f in FEATURES:
        for k in trajs: trajs[k][f] = []  # list of arrays length N_T

    per_episode_meta = []
    for r in SUMMARIES:
        cls = classify(r)
        if cls == "OTHER": continue
        ps_path = os.path.join(str(DATA_DIR), "per_sample",
                               os.path.basename(r["csv_path"]).replace(".csv", ".per_sample.json"))
        if not os.path.exists(ps_path): continue
        ps = json.load(open(ps_path))
        ci = r["contact_idx_active"]
        if ci is None: continue
        t_active = np.asarray(ps["t_s"], dtype=float)
        t_contact = t_active[ci]
        t_rel = t_active - t_contact  # 0 at contact

        for f in FEATURES:
            arr = ps[f] if f in ps else None
            if arr is None: continue
            x_aligned = resample(t_rel, arr, T_GRID)
            trajs[cls][f].append(x_aligned)

        per_episode_meta.append({"csv": os.path.basename(r["csv_path"]),
                                 "object": r["object"], "outcome": r["outcome"],
                                 "class": cls, "final_z_drop_mm": r["final_z_drop_mm"]})

    # Stack & compute per-class quantiles per timepoint
    summary = {"t_s": T_GRID.tolist(), "n": {k: len(trajs[k]["fz_t"]) for k in trajs}, "feats": {}}
    for f in FEATURES:
        summary["feats"][f] = {}
        for cls in trajs:
            stk = np.stack(trajs[cls][f]) if trajs[cls][f] else np.zeros((0, N_T))
            with np.errstate(all="ignore"):
                p = {q: np.nanpercentile(stk, q, axis=0) if stk.size else np.full(N_T, np.nan)
                     for q in (10, 25, 50, 75, 90)}
            summary["feats"][f][cls] = {f"p{q}": v.tolist() for q, v in p.items()}

    json.dump(summary, open(os.path.join(str(DATA_DIR), "aligned_summary.json"), "w"))
    print(f"aligned: GOLD n={summary['n']['GOLD']} | FAIL n={summary['n']['FAIL']} | AUTO_SEATED n={summary['n']['AUTO_SEATED']}")
    print(f"window: t in [{T_PRE:+.1f}, {T_POST:+.1f}] s, dt={DT*1000:.0f}ms, N={N_T}")

    # Print key feature divergence: where does FAIL p50 leave GOLD [p25, p75] band?
    print(f"\n{'feature':14s}  divergence_t(s)  GOLD@div   FAIL@div    GOLD_band@div")
    for f in ["fz_t", "z_drop_mm", "F_lat_base", "dz_dt_mm_s", "tilt_deg"]:
        gp25 = np.array(summary["feats"][f]["GOLD"]["p25"])
        gp75 = np.array(summary["feats"][f]["GOLD"]["p75"])
        gp50 = np.array(summary["feats"][f]["GOLD"]["p50"])
        fp50 = np.array(summary["feats"][f]["FAIL"]["p50"])
        # find first t > 0 where fp50 falls outside [gp25, gp75]
        post = np.where(T_GRID >= 0.0)[0]
        div_idx = None
        for i in post:
            if not np.isfinite(fp50[i]) or not np.isfinite(gp25[i]): continue
            if fp50[i] < gp25[i] or fp50[i] > gp75[i]:
                div_idx = i; break
        if div_idx is None:
            print(f"  {f:14s}     (never diverges)")
            continue
        print(f"  {f:14s}     {T_GRID[div_idx]:>+5.2f}        "
              f"{gp50[div_idx]:>+7.2f}    {fp50[div_idx]:>+7.2f}    "
              f"[{gp25[div_idx]:>+7.2f}, {gp75[div_idx]:>+7.2f}]")

    # Save per-episode rich data for downstream (per-failure divergence times)
    json.dump({"meta": per_episode_meta, "t_s": T_GRID.tolist(), "n": summary["n"]},
              open(os.path.join(str(DATA_DIR), "aligned_index.json"), "w"))


if __name__ == "__main__":
    main()
