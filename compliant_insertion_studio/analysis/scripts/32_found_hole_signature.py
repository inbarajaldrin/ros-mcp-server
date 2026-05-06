"""
Found Hole signature analysis.

Filters to the 10 clean GUIDED demos (csv_final == DONE AND
hole_observed_operator.source == 'fsm_guided_sigusr1'). Computes tool-frame
features only, aggregates across demos, and ranks them by their ability to
discriminate the SIGUSR1 moment from the rest of the GUIDED segment.

Tool-frame features tested (all direction-invariant by construction):
  - tilt_deg               : EE-Z vs world-Z angle (magnitude only, frame-free)
  - F_lat_tool             : sqrt(fx^2 + fy^2)  in tool frame
  - rcop_mag_m             : |(-Ty, Tx)| / Fz   center-of-pressure radius (peg face)
  - fz_smooth              : smoothed normal load
  - vz                     : world-frame Z velocity (sign-invariant magnitude in
                              practice since EE is roughly face-down)

Per-demo, four time-anchored snapshots:
  - baseline: median over [t_contact, t_sigusr1 - 1.0 s]    (operator-search steady state)
  - pre_300ms: median over [t_sigusr1 - 0.3 s, t_sigusr1]   (just before mark)
  - at_event: value at t_sigusr1
  - post_300ms: median over [t_sigusr1, t_sigusr1 + 0.3 s]  (after mark)

Output: stats table + cross-demo summary; basis for the predicate in 33.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _paths import DATA_DIR, LOG_DIR  # noqa: E402

FEATURES_DIR = DATA_DIR / "guided_features"
OUT_PATH = DATA_DIR / "found_hole_signature.json"

VARIATION_OF = {
    "insert_u_orange_20260506_040446": "C_pos_y_10mm",
    "insert_u_orange_20260506_040628": "C_pos_y_10mm",
    "insert_u_orange_20260506_042120": "C_pos_y_10mm",
    "insert_u_orange_20260506_042232": "D_neg_y_10mm",
    "insert_u_orange_20260506_043107": "D_neg_y_10mm",
    "insert_u_orange_20260506_043221": "D_neg_y_10mm",
    "insert_u_orange_20260506_043324": "D_neg_y_10mm",
    "insert_u_orange_20260506_043426": "A_pos_x_10mm",
    "insert_u_orange_20260506_043529": "A_pos_x_10mm",
    "insert_u_orange_20260506_043633": "A_pos_x_10mm",
}


def _is_clean_demo(meta_path: str) -> bool:
    m = json.load(open(meta_path))
    h = m.get("hole_observed_operator")
    if not h or h.get("source") != "fsm_guided_sigusr1":
        return False
    csv_p = meta_path.replace(".meta.json", ".csv")
    if not os.path.exists(csv_p):
        return False
    last = None
    for r in csv.DictReader(open(csv_p)):
        last = r
    return last is not None and last.get("phase") == "DONE"


def _percentile_band(arr):
    return {
        "p5":  float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _per_demo_snapshots(features: dict) -> dict:
    """Compute baseline/pre/at/post snapshots for each tool-frame feature."""
    g = features["guided"]
    s = features["summary"]
    dt = s["dt_s"]
    t_s = np.array(g["t_s"])
    t_event = s["t_sigusr1_s"]
    t_contact = s["t_contact_s"]

    # Window indices
    i_event = int(np.argmin(np.abs(t_s - t_event)))
    i_pre300 = int(np.argmin(np.abs(t_s - (t_event - 0.3))))
    i_baseline_end = int(np.argmin(np.abs(t_s - (t_event - 1.0))))
    i_baseline_start = 0  # first GUIDED sample = t_contact

    # Tool-frame, frame-invariant features
    tilt   = np.array(g["tilt_deg"])
    flat_t = np.array(g["F_lat_tool"])
    rcop_m = 1000 * np.array(g["rcop_mag_m"])
    fz_s   = np.array(g["fz_smooth"])
    vz     = np.array(g["vz"])

    # Need post_300ms window: peek into descent slice
    d = features.get("descent", {})
    if d:
        t_d = np.array(d["t_s"])
        # For post_300ms we need [t_event, t_event+0.3]; descent starts at t_event
        i_post_end = int(np.argmin(np.abs(t_d - (t_event + 0.3))))
        post_tilt = np.array(d["tilt_deg"])[: max(1, i_post_end)]
        post_flat = np.array(d["F_lat_tool"])[: max(1, i_post_end)]
        post_rcop = 1000 * np.array(d["rcop_mag_m"])[: max(1, i_post_end)]
        post_fz   = np.array(d["fz_smooth"])[: max(1, i_post_end)]
        post_vz   = np.array(d["vz"])[: max(1, i_post_end)]
    else:
        post_tilt = post_flat = post_rcop = post_fz = post_vz = np.array([np.nan])

    def med_safe(arr):
        return float(np.nanmedian(arr)) if len(arr) else float("nan")

    def feat_block(arr_g, arr_post):
        baseline = arr_g[i_baseline_start:max(1, i_baseline_end)]
        pre300   = arr_g[i_pre300:i_event + 1]
        at_event = arr_g[i_event]
        rolling_peak_1s = float(np.max(arr_g[max(0, i_event - int(1.0 / dt)): i_event + 1]))
        rolling_min_1s  = float(np.min(arr_g[max(0, i_event - int(1.0 / dt)): i_event + 1]))
        return {
            "baseline_median": med_safe(baseline),
            "baseline_p25": float(np.percentile(baseline, 25)) if len(baseline) else float("nan"),
            "baseline_p75": float(np.percentile(baseline, 75)) if len(baseline) else float("nan"),
            "pre300_median": med_safe(pre300),
            "at_event": float(at_event),
            "post300_median": med_safe(arr_post),
            "rolling_peak_1s_pre": rolling_peak_1s,
            "rolling_min_1s_pre": rolling_min_1s,
            "drop_from_peak_1s": rolling_peak_1s - float(at_event),
            "rise_from_min_1s":  float(at_event) - rolling_min_1s,
        }

    return {
        "basename": features["basename"],
        "variation": VARIATION_OF.get(features["basename"], "?"),
        "guided_dur_s": s["guided_dur_s"],
        "tilt_deg":      feat_block(tilt,   post_tilt),
        "F_lat_tool_N":  feat_block(flat_t, post_flat),
        "rcop_mag_mm":   feat_block(rcop_m, post_rcop),
        "fz_smooth_N":   feat_block(fz_s,   post_fz),
        "vz_m_s":        feat_block(vz,     post_vz),
    }


def _rolling_baseline_windows(features: dict, win_s: float = 0.3) -> dict:
    """For each feature, gather all rolling 300ms-window medians from the
    portion of GUIDED that excludes the [event-300ms, event] window. These are
    the 'negative class' for AUC computation: rolling values during operator search
    NOT at the labeled hole moment."""
    g = features["guided"]
    s = features["summary"]
    dt = s["dt_s"]
    t_s = np.array(g["t_s"])
    t_event = s["t_sigusr1_s"]
    win_n = max(1, int(round(win_s / dt)))
    i_event = int(np.argmin(np.abs(t_s - t_event)))
    exclude_start = max(0, i_event - win_n)

    arrs = {
        "tilt_deg":     np.array(g["tilt_deg"]),
        "F_lat_tool_N": np.array(g["F_lat_tool"]),
        "rcop_mag_mm":  1000 * np.array(g["rcop_mag_m"]),
        "fz_smooth_N":  np.array(g["fz_smooth"]),
        "vz_m_s":       np.array(g["vz"]),
    }

    # Derived discontinuity features: drop_from_peak_1s, rise_from_min_1s
    out = {}
    for name, arr in arrs.items():
        # Sliding rolling median over win_n
        if len(arr) < win_n + 2:
            out[name + "_rolling_med"] = []
            out[name + "_drop_from_peak_1s"] = []
            out[name + "_rise_from_min_1s"] = []
            continue
        # Naive O(n*win) rolling — fine for ~1000 samples per demo
        rolling_med = np.array([np.median(arr[max(0, i - win_n):i + 1]) for i in range(len(arr))])

        # rolling 1s peak/min ending at i, excluding i+1..end
        peak_n = int(round(1.0 / dt))
        rolling_peak = np.array([np.max(arr[max(0, i - peak_n):i + 1]) for i in range(len(arr))])
        rolling_min  = np.array([np.min(arr[max(0, i - peak_n):i + 1]) for i in range(len(arr))])
        drop_from_peak = rolling_peak - arr
        rise_from_min  = arr - rolling_min

        # Exclude indices within [exclude_start, i_event]
        mask = np.ones(len(arr), dtype=bool)
        mask[exclude_start:i_event + 1] = False

        out[name + "_rolling_med"]      = rolling_med[mask].tolist()
        out[name + "_drop_from_peak_1s"] = drop_from_peak[mask].tolist()
        out[name + "_rise_from_min_1s"]  = rise_from_min[mask].tolist()

    return out


def _auc(pos: np.ndarray, neg: np.ndarray, higher_is_event=True) -> float:
    """Mann-Whitney U / N1*N2 = AUC. Higher value of feature → event present."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    if not higher_is_event:
        pos, neg = -pos, -neg
    combined = np.concatenate([pos, neg])
    ranks = combined.argsort().argsort() + 1
    rsum_pos = ranks[: len(pos)].sum()
    n1, n2 = len(pos), len(neg)
    u = rsum_pos - n1 * (n1 + 1) / 2.0
    return float(u / (n1 * n2))


def main():
    # 1. Find clean demos
    metas = sorted([
        os.path.join(str(LOG_DIR), f"{bn}.meta.json")
        for bn in VARIATION_OF.keys()
    ])
    clean_demos = []
    for mp in metas:
        if not os.path.exists(mp):
            continue
        if _is_clean_demo(mp):
            clean_demos.append(os.path.basename(mp).replace(".meta.json", ""))
    print(f"Clean demos: {len(clean_demos)}")
    for bn in clean_demos:
        print(f"  {bn}  ({VARIATION_OF[bn]})")

    if len(clean_demos) < 5:
        print("Too few clean demos for analysis.")
        sys.exit(1)

    # 2. Load extracted features for each
    snapshots = []
    rolling_pools = {}  # feature_name -> list across all demos
    pos_pools = {}      # feature_name -> per-demo at-event values
    for bn in clean_demos:
        fp = FEATURES_DIR / f"{bn}.features.json"
        if not fp.exists():
            print(f"  MISSING features json for {bn}")
            continue
        feats = json.load(open(fp))
        snap = _per_demo_snapshots(feats)
        snapshots.append(snap)

        # Pool rolling baseline values for AUC
        roll = _rolling_baseline_windows(feats)
        for k, v in roll.items():
            rolling_pools.setdefault(k, []).extend(v)

        # Pos class: at-event values + derived discontinuity at event
        for fname in ["tilt_deg", "F_lat_tool_N", "rcop_mag_mm", "fz_smooth_N", "vz_m_s"]:
            pos_pools.setdefault(fname + "_at_event", []).append(snap[fname]["at_event"])
            pos_pools.setdefault(fname + "_drop_from_peak_1s", []).append(snap[fname]["drop_from_peak_1s"])
            pos_pools.setdefault(fname + "_rise_from_min_1s", []).append(snap[fname]["rise_from_min_1s"])

    # 3. Print per-demo signature table (tool-frame features only)
    print()
    print("=== Per-demo at-SIGUSR1 snapshot (tool-frame) ===")
    hdr = (f"{'demo':<32s}{'var':<8s} | "
           f"{'tilt°':>6s} {'tilt_drop_1s°':>13s} | "
           f"{'F_lat_N':>8s} {'F_drop_1s_N':>11s} | "
           f"{'rcop_mm':>7s} {'rcop_drop_mm':>12s} | "
           f"{'vz_mm/s':>8s}")
    print(hdr)
    print("-" * len(hdr))
    for s in snapshots:
        v = s["variation"].split("_")[0]
        print(f"{s['basename'][-15:]:<32s}{v:<8s} | "
              f"{s['tilt_deg']['at_event']:>6.2f} {s['tilt_deg']['drop_from_peak_1s']:>13.2f} | "
              f"{s['F_lat_tool_N']['at_event']:>8.2f} {s['F_lat_tool_N']['drop_from_peak_1s']:>11.2f} | "
              f"{s['rcop_mag_mm']['at_event']:>7.2f} {s['rcop_mag_mm']['drop_from_peak_1s']:>12.2f} | "
              f"{1000*s['vz_m_s']['at_event']:>8.2f}")

    # 4. AUC for each feature (event vs rolling baseline)
    # For each feature_pos, find the corresponding _rolling_med or derived rolling pool
    print()
    print("=== Cross-demo AUC (at-event vs rolling-baseline windows) ===")
    print(f"{'feature':<40s} {'pos n':>6s} {'neg n':>7s}  AUC (higher_is_event)  AUC (lower_is_event)")
    rankings = []
    pos_neg_pairs = [
        ("tilt_deg_at_event",          "tilt_deg_rolling_med"),
        ("tilt_deg_drop_from_peak_1s", "tilt_deg_drop_from_peak_1s"),
        ("F_lat_tool_N_at_event",      "F_lat_tool_N_rolling_med"),
        ("F_lat_tool_N_drop_from_peak_1s", "F_lat_tool_N_drop_from_peak_1s"),
        ("rcop_mag_mm_at_event",       "rcop_mag_mm_rolling_med"),
        ("rcop_mag_mm_drop_from_peak_1s", "rcop_mag_mm_drop_from_peak_1s"),
        ("fz_smooth_N_at_event",       "fz_smooth_N_rolling_med"),
        ("fz_smooth_N_drop_from_peak_1s", "fz_smooth_N_drop_from_peak_1s"),
        ("vz_m_s_at_event",            "vz_m_s_rolling_med"),
        ("vz_m_s_rise_from_min_1s",    "vz_m_s_rise_from_min_1s"),
    ]
    for pname, nname in pos_neg_pairs:
        pos = np.array(pos_pools.get(pname, []), dtype=float)
        neg = np.array(rolling_pools.get(nname, []), dtype=float)
        pos = pos[~np.isnan(pos)]
        neg = neg[~np.isnan(neg)]
        auc_hi = _auc(pos, neg, higher_is_event=True)
        auc_lo = _auc(pos, neg, higher_is_event=False)
        rankings.append((pname, len(pos), len(neg), auc_hi, auc_lo))
        print(f"{pname:<40s} {len(pos):>6d} {len(neg):>7d}  {auc_hi:>10.3f}             {auc_lo:>10.3f}")

    # 5. Per-feature event distribution stats (tells us threshold range)
    print()
    print("=== At-event feature distributions (across 10 clean demos) ===")
    for fname in ["tilt_deg", "F_lat_tool_N", "rcop_mag_mm", "fz_smooth_N", "vz_m_s"]:
        for kind in ["at_event", "drop_from_peak_1s", "rise_from_min_1s"]:
            key = fname + "_" + kind
            arr = pos_pools.get(key, [])
            if not arr:
                continue
            arr = np.array(arr, dtype=float)
            arr = arr[~np.isnan(arr)]
            if len(arr) == 0:
                continue
            print(f"  {key:<40s} {_percentile_band(arr)}")

    out = {
        "clean_demos": clean_demos,
        "variations": [VARIATION_OF[bn] for bn in clean_demos],
        "snapshots": snapshots,
        "auc_rankings": [
            {"feature": p, "n_pos": n_p, "n_neg": n_n, "auc_higher_is_event": a_hi, "auc_lower_is_event": a_lo}
            for p, n_p, n_n, a_hi, a_lo in rankings
        ],
        "pos_pools": {k: list(v) for k, v in pos_pools.items()},
    }
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
