"""
Phase G — Search director derivation.

Re-analyzes the 10 clean GUIDED demos focused on the SEARCH segment only
(Contact → rim-cross moment, where rim-cross = first time |fz_smoothed|
drops below 3N for sustained 0.3s after having been above 4N).

Goal: identify a tool-frame F/T-derived signal whose direction predicts the
operator's drag direction during search. The autonomous controller will use
this signal to actively command a lateral wrench in the same direction the
operator's hand was pulling.

Hypothesis: when peg is pressed on rim, the wrench at TCP shows a contact
at off-axis location. r_cop_tool = (-Ty/Fz, Tx/Fz) points from peg-axis
toward the contact point on the peg face. Rotated to base frame and negated,
-r_cop_base points AWAY from the rim, which is geometrically toward the open
side of the chamfer (the hole). Operator's drag direction should align.

Analysis:
  1. Slice each demo's CSV into [t_contact, t_rim_cross]. Filter out samples
     where |fz_smoothed| < 4N (where r_cop is meaningless).
  2. Compute drag direction (smoothed v_xy in base frame) and -r_cop in base
     frame at each tick.
  3. Compute alignment dot(drag_unit, -r_cop_unit) per tick, and aggregate.
  4. Cross-demo: median + p5/p95.
  5. If alignment is consistently positive across demos, we have the director.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).parent))
from _paths import DATA_DIR, LOG_DIR  # noqa: E402

CLEAN_DEMOS_AND_VARIATIONS = [
    ("insert_u_orange_20260506_040446", "C_pos_y_10mm"),
    ("insert_u_orange_20260506_040628", "C_pos_y_10mm"),
    ("insert_u_orange_20260506_042120", "C_pos_y_10mm"),
    ("insert_u_orange_20260506_042232", "D_neg_y_10mm"),
    ("insert_u_orange_20260506_043107", "D_neg_y_10mm"),
    ("insert_u_orange_20260506_043221", "D_neg_y_10mm"),
    ("insert_u_orange_20260506_043324", "D_neg_y_10mm"),
    ("insert_u_orange_20260506_043426", "A_pos_x_10mm"),
    ("insert_u_orange_20260506_043529", "A_pos_x_10mm"),
    ("insert_u_orange_20260506_043633", "A_pos_x_10mm"),
]

OUT_PATH = DATA_DIR / "search_director_analysis.json"


def _smooth(x: np.ndarray, window_s: float, dt: float) -> np.ndarray:
    n = max(1, int(round(window_s / dt)))
    if n <= 1 or len(x) < n:
        return x.copy()
    kernel = np.ones(n) / n
    return np.convolve(x, kernel, mode="same")


def _find_rim_cross(t_s: np.ndarray, fz_smoothed: np.ndarray,
                    rim_high=4.0, rim_low=3.0,
                    off_sustain_s=0.3, recent_window_s=2.5,
                    dt=0.01) -> int | None:
    """Replicate the v4 detector logic (sample-based) to find the rim-cross
    sample index. Returns None if predicate doesn't fire."""
    n_off_sustain = max(1, int(round(off_sustain_s / dt)))
    n_recent = max(1, int(round(recent_window_s / dt)))
    abs_fz = np.abs(fz_smoothed)
    on_rim = abs_fz > rim_high
    off_rim = abs_fz < rim_low
    off_run = 0
    for i in range(len(fz_smoothed)):
        if off_rim[i]:
            off_run += 1
        else:
            off_run = 0
        if off_run >= n_off_sustain:
            lo = max(0, i - n_recent)
            hi = max(0, i - n_off_sustain)
            if hi > lo and np.any(on_rim[lo:hi]):
                return i - n_off_sustain + 1
    return None


def _process_demo(bn: str, variation: str) -> dict | None:
    csv_p = LOG_DIR / f"{bn}.csv"
    meta_p = LOG_DIR / f"{bn}.meta.json"
    if not csv_p.exists() or not meta_p.exists():
        return None
    meta = json.load(open(meta_p))
    bias = meta.get("post_zero_bias") or {}
    bx = bias.get("Fx", 0.0); by = bias.get("Fy", 0.0); bz = bias.get("Fz", 0.0)
    btx = bias.get("Tx", 0.0); bty = bias.get("Ty", 0.0); btz = bias.get("Tz", 0.0)

    rows = [r for r in csv.DictReader(open(csv_p)) if r.get("phase") == "ACTIVE"]
    if len(rows) < 100:
        return None

    def col(k): return np.array([float(r[k]) for r in rows], dtype=float)

    t_s = col("t_s")
    tcp_x = col("tcp_x"); tcp_y = col("tcp_y"); tcp_z = col("tcp_z")
    qx = col("tcp_qx"); qy = col("tcp_qy"); qz = col("tcp_qz"); qw = col("tcp_qw")
    fx = col("fx") - bx; fy = col("fy") - by; fz = col("fz") - bz
    tx = col("tx") - btx; ty = col("ty") - bty
    quats = np.column_stack([qx, qy, qz, qw])

    dt = float(np.median(np.diff(t_s)))
    if not (0.005 < dt < 0.05):
        return None

    fz_s = _smooth(fz, 0.1, dt)
    abs_fz_s = np.abs(fz_s)

    # Find Contact: first sustained fz > 3N for 100ms after a 1s grace
    n_sustain = max(1, int(round(0.1 / dt)))
    grace_end_t = t_s[0] + 1.0
    i_contact = None
    for i in range(len(abs_fz_s) - n_sustain):
        if t_s[i] < grace_end_t:
            continue
        if all(abs_fz_s[i + j] > 3.0 for j in range(n_sustain)):
            i_contact = i
            break
    if i_contact is None:
        return None

    # Find rim-cross via v4-style detector applied from i_contact onward
    i_rim_offset = _find_rim_cross(t_s[i_contact:], fz_s[i_contact:],
                                    dt=dt)
    if i_rim_offset is None:
        return None
    i_rim = i_contact + i_rim_offset

    # Compute velocities (centered diff, smoothed 200ms)
    def cdiff(arr):
        v = np.zeros_like(arr)
        v[1:-1] = (arr[2:] - arr[:-2]) / (2 * dt)
        v[0] = (arr[1] - arr[0]) / dt
        v[-1] = (arr[-1] - arr[-2]) / dt
        return v

    vx = _smooth(cdiff(tcp_x), 0.2, dt)
    vy = _smooth(cdiff(tcp_y), 0.2, dt)
    drag_speed = np.hypot(vx, vy)
    eps = 1e-6
    safe_drag = drag_speed > 5e-4  # 0.5 mm/s

    # r_cop in tool frame (m). Guard against tiny |fz|.
    fz_for_cop = np.where(np.abs(fz_s) < 0.5, np.copysign(0.5, fz_s + eps), fz_s)
    rcop_x_tool = (-ty) / fz_for_cop  # m
    rcop_y_tool = ( tx) / fz_for_cop  # m
    rcop_mag = np.hypot(rcop_x_tool, rcop_y_tool)

    # Rotate -r_cop tool-XY vector to BASE frame using TCP quat
    n = len(rows)
    minus_rcop_tool = np.column_stack([-rcop_x_tool, -rcop_y_tool, np.zeros(n)])
    rots = R.from_quat(quats)
    minus_rcop_base = rots.apply(minus_rcop_tool)
    minus_rcop_x_base = minus_rcop_base[:, 0]
    minus_rcop_y_base = minus_rcop_base[:, 1]
    minus_rcop_xy_norm = np.hypot(minus_rcop_x_base, minus_rcop_y_base)
    minus_rcop_ux = np.where(minus_rcop_xy_norm > eps,
                              minus_rcop_x_base / np.maximum(minus_rcop_xy_norm, eps), 0.0)
    minus_rcop_uy = np.where(minus_rcop_xy_norm > eps,
                              minus_rcop_y_base / np.maximum(minus_rcop_xy_norm, eps), 0.0)

    # Drag direction unit (base XY)
    drag_ux = np.where(safe_drag, vx / np.maximum(drag_speed, eps), 0.0)
    drag_uy = np.where(safe_drag, vy / np.maximum(drag_speed, eps), 0.0)

    # Per-tick alignment
    align = drag_ux * minus_rcop_ux + drag_uy * minus_rcop_uy

    # Filter to the SEARCH SEGMENT [i_contact, i_rim] AND |fz|>4N
    seg = slice(i_contact, i_rim)
    mask_search = np.zeros(n, dtype=bool)
    mask_search[seg] = True
    mask_high_fz = abs_fz_s > 4.0
    mask_safe_drag = safe_drag
    mask = mask_search & mask_high_fz & mask_safe_drag

    align_filt = align[mask]
    rcop_mag_filt_mm = 1000 * rcop_mag[mask]
    drag_speed_filt = drag_speed[mask]
    abs_fz_filt = abs_fz_s[mask]

    # Where the operator went from contact to rim-cross
    contact_xy = np.array([tcp_x[i_contact], tcp_y[i_contact]])
    rim_cross_xy = np.array([tcp_x[i_rim], tcp_y[i_rim]])
    delta_xy = rim_cross_xy - contact_xy
    dist_traveled = float(np.linalg.norm(delta_xy))
    direction_to_hole_unit = delta_xy / max(eps, dist_traveled)

    # Per-tick alignment of -r_cop with the OVERALL direction-to-hole vector
    # (sanity check: does -r_cop on average point toward where peg ended up?)
    align_to_hole = (minus_rcop_ux * direction_to_hole_unit[0]
                     + minus_rcop_uy * direction_to_hole_unit[1])
    align_to_hole_filt = align_to_hole[mask]

    return {
        "basename": bn,
        "variation": variation,
        "i_contact": int(i_contact),
        "i_rim": int(i_rim),
        "search_dur_s": float(t_s[i_rim] - t_s[i_contact]),
        "search_dist_m": dist_traveled,
        "search_speed_mean_mm_s": float(1000 * np.mean(drag_speed_filt)) if len(drag_speed_filt) else float("nan"),
        "search_speed_max_mm_s":  float(1000 * np.max(drag_speed_filt)) if len(drag_speed_filt) else float("nan"),
        "abs_fz_in_search_p5":  float(np.percentile(abs_fz_filt, 5)) if len(abs_fz_filt) else float("nan"),
        "abs_fz_in_search_med": float(np.median(abs_fz_filt)) if len(abs_fz_filt) else float("nan"),
        "abs_fz_in_search_p95": float(np.percentile(abs_fz_filt, 95)) if len(abs_fz_filt) else float("nan"),
        "rcop_mag_mm_p5":  float(np.percentile(rcop_mag_filt_mm, 5)) if len(rcop_mag_filt_mm) else float("nan"),
        "rcop_mag_mm_med": float(np.median(rcop_mag_filt_mm)) if len(rcop_mag_filt_mm) else float("nan"),
        "rcop_mag_mm_p95": float(np.percentile(rcop_mag_filt_mm, 95)) if len(rcop_mag_filt_mm) else float("nan"),
        "n_high_fz_samples_in_search": int(mask.sum()),
        "n_search_samples_total": int(mask_search.sum()),
        "frac_high_fz_in_search": float(mask.sum() / max(1, mask_search.sum())),
        "align_drag_vs_minus_rcop_p5":  float(np.percentile(align_filt, 5)) if len(align_filt) else float("nan"),
        "align_drag_vs_minus_rcop_med": float(np.median(align_filt)) if len(align_filt) else float("nan"),
        "align_drag_vs_minus_rcop_p95": float(np.percentile(align_filt, 95)) if len(align_filt) else float("nan"),
        "align_drag_vs_minus_rcop_mean": float(np.mean(align_filt)) if len(align_filt) else float("nan"),
        "align_minus_rcop_to_hole_med": float(np.median(align_to_hole_filt)) if len(align_to_hole_filt) else float("nan"),
        "align_minus_rcop_to_hole_mean": float(np.mean(align_to_hole_filt)) if len(align_to_hole_filt) else float("nan"),
        # Force magnitude operator applied during drag (informs F_search gain)
        "F_lat_applied_med_N": float(np.median(np.hypot(fx[mask], fy[mask]))) if mask.sum() else float("nan"),
        "F_lat_applied_p95_N": float(np.percentile(np.hypot(fx[mask], fy[mask]), 95)) if mask.sum() else float("nan"),
    }


def main():
    rows = []
    print(f"{'demo':<15s}{'var':<8s} {'search_s':>9s} {'dist_mm':>8s} {'spd_mm/s':>9s} "
          f"{'fz_med':>7s} {'rcop_med':>9s} | {'align_drag_vs_-rcop':>22s} | {'align_-rcop→hole':>18s}")
    print("-" * 130)
    for bn, var in CLEAN_DEMOS_AND_VARIATIONS:
        try:
            r = _process_demo(bn, var)
        except Exception as e:
            print(f"  FAIL {bn}: {e}")
            continue
        if r is None:
            print(f"  SKIP {bn}")
            continue
        rows.append(r)
        v = var.split("_")[0]
        print(f"{bn[-15:]:<15s}{v:<8s} "
              f"{r['search_dur_s']:>9.2f} "
              f"{1000*r['search_dist_m']:>8.1f} "
              f"{r['search_speed_mean_mm_s']:>9.2f} "
              f"{r['abs_fz_in_search_med']:>7.2f} "
              f"{r['rcop_mag_mm_med']:>9.2f} | "
              f"med={r['align_drag_vs_minus_rcop_med']:>+5.2f} "
              f"mean={r['align_drag_vs_minus_rcop_mean']:>+5.2f} | "
              f"med={r['align_minus_rcop_to_hole_med']:>+5.2f} "
              f"mean={r['align_minus_rcop_to_hole_mean']:>+5.2f}")

    print()
    print("=== Cross-demo aggregates ===")
    aligns_drag = np.array([r['align_drag_vs_minus_rcop_mean'] for r in rows])
    aligns_hole = np.array([r['align_minus_rcop_to_hole_mean'] for r in rows])
    speed_med = np.array([r['search_speed_mean_mm_s'] for r in rows])
    flat_p95 = np.array([r['F_lat_applied_p95_N'] for r in rows])
    durs = np.array([r['search_dur_s'] for r in rows])
    dists = np.array([1000 * r['search_dist_m'] for r in rows])

    print(f"align(drag, -r_cop):     mean={np.mean(aligns_drag):+.3f}  median={np.median(aligns_drag):+.3f}  p5={np.percentile(aligns_drag,5):+.3f} p95={np.percentile(aligns_drag,95):+.3f}")
    print(f"align(-r_cop, →hole):    mean={np.mean(aligns_hole):+.3f}  median={np.median(aligns_hole):+.3f}  p5={np.percentile(aligns_hole,5):+.3f} p95={np.percentile(aligns_hole,95):+.3f}")
    print(f"search duration (s):     median={np.median(durs):.2f}  p5={np.percentile(durs,5):.2f} p95={np.percentile(durs,95):.2f}")
    print(f"search distance (mm):    median={np.median(dists):.1f}  p5={np.percentile(dists,5):.1f} p95={np.percentile(dists,95):.1f}")
    print(f"drag speed mean (mm/s):  median={np.median(speed_med):.2f}  p5={np.percentile(speed_med,5):.2f} p95={np.percentile(speed_med,95):.2f}")
    print(f"F_lat applied p95 (N):   median={np.median(flat_p95):.2f}  p5={np.percentile(flat_p95,5):.2f} p95={np.percentile(flat_p95,95):.2f}")

    out = {"demos": rows,
           "aggregates": {
               "align_drag_vs_minus_rcop_mean":  float(np.mean(aligns_drag)),
               "align_drag_vs_minus_rcop_median": float(np.median(aligns_drag)),
               "align_minus_rcop_to_hole_mean":  float(np.mean(aligns_hole)),
               "align_minus_rcop_to_hole_median": float(np.median(aligns_hole)),
               "search_dur_s_median": float(np.median(durs)),
               "search_dist_mm_median": float(np.median(dists)),
               "drag_speed_mm_s_median": float(np.median(speed_med)),
               "F_lat_applied_p95_median": float(np.median(flat_p95)),
           }}
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
