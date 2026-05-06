"""
Phase G — Comprehensive F_lat-direction analysis (per user's hint).

Re-analyzes the 10 GUIDED demos to determine: which F/T-derived signal direction
most reliably points toward the hole during the search segment?

Candidate signals (all in base_link frame):
  S1: -r_cop = -((-Ty/Fz, Tx/Fz)) — original "center of pressure" approach
  S2: -F_lat_sensed = (-Fx, -Fy) — sensed lateral force direction reversed
  S3: +F_lat_sensed = (+Fx, +Fy) — sensed lateral force as-is
  S4: peg velocity = drag direction (smoothed)
  S5: -∇|r_cop| projected onto motion plane — gradient (closer to hole = smaller r_cop)
  S6: -∇|F_lat| — gradient (closer to hole = ?)

For each, compute alignment with:
  T1: instantaneous drag direction (operator's choice at that tick)
  T2: contact-to-rim_cross overall vector (hole direction)
  T3: rim_cross_xy - tcp_xy (per-tick, dynamic)

Also look at TEMPORAL EVOLUTION: does any signal STRENGTHEN as peg approaches
rim-cross (last 1s of search vs first 1s)?
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).parent))
from _paths import DATA_DIR, LOG_DIR  # noqa: E402

CLEAN_DEMOS = [
    "insert_u_orange_20260506_040446",
    "insert_u_orange_20260506_040628",
    "insert_u_orange_20260506_042120",
    "insert_u_orange_20260506_042232",
    "insert_u_orange_20260506_043107",
    "insert_u_orange_20260506_043221",
    "insert_u_orange_20260506_043324",
    "insert_u_orange_20260506_043426",
    "insert_u_orange_20260506_043529",
    "insert_u_orange_20260506_043633",
]


def _process(bn: str) -> dict | None:
    csvp = LOG_DIR / f"{bn}.csv"
    metap = LOG_DIR / f"{bn}.meta.json"
    if not csvp.exists():
        return None
    m = json.load(open(metap))
    bias = m.get("post_zero_bias", {})
    rows = [r for r in csv.DictReader(open(csvp)) if r.get("phase") == "ACTIVE"]
    if len(rows) < 100:
        return None

    def col(k):
        return np.array([float(r[k]) for r in rows], dtype=float)

    ts = col("t_s")
    tcp_x = col("tcp_x"); tcp_y = col("tcp_y")
    fx_t = col("fx") - bias.get("Fx", 0)
    fy_t = col("fy") - bias.get("Fy", 0)
    fz_t = col("fz") - bias.get("Fz", 0)
    tx_t = col("tx") - bias.get("Tx", 0)
    ty_t = col("ty") - bias.get("Ty", 0)
    tz_t = col("tz") - bias.get("Tz", 0)
    qx = col("tcp_qx"); qy = col("tcp_qy"); qz = col("tcp_qz"); qw = col("tcp_qw")

    quats = np.column_stack([qx, qy, qz, qw])
    rots = R.from_quat(quats)
    F_b = rots.apply(np.column_stack([fx_t, fy_t, fz_t]))
    T_b = rots.apply(np.column_stack([tx_t, ty_t, tz_t]))
    fx_b = F_b[:, 0]; fy_b = F_b[:, 1]; fz_b = F_b[:, 2]
    tx_b = T_b[:, 0]; ty_b = T_b[:, 1]
    fz_b_smooth = (np.convolve(np.abs(fz_b), np.ones(10)/10, mode="same") * np.sign(fz_b))

    # Velocity
    def cdiff(arr):
        v = np.zeros_like(arr); v[1:-1] = (arr[2:] - arr[:-2]) / 0.02
        v[0] = (arr[1] - arr[0]) / 0.01
        v[-1] = (arr[-1] - arr[-2]) / 0.01
        return v
    vx = np.convolve(cdiff(tcp_x), np.ones(20)/20, mode="same")
    vy = np.convolve(cdiff(tcp_y), np.ones(20)/20, mode="same")
    spd = np.hypot(vx, vy)

    # Find Contact + rim-cross
    abs_fz_b = np.abs(fz_b_smooth)
    n = len(rows)
    i_contact = None
    for i in range(n - 10):
        if ts[i] - ts[0] < 1.0: continue
        if all(abs_fz_b[i + j] > 3 for j in range(10)):
            i_contact = i; break
    if i_contact is None:
        return None
    n_off = 30; n_recent = 250
    off_run = 0; i_rim = None
    for i in range(i_contact, n):
        if abs_fz_b[i] < 3: off_run += 1
        else: off_run = 0
        if off_run >= n_off:
            lo = max(0, i - n_recent); hi = max(0, i - n_off)
            if hi > lo and np.any(abs_fz_b[lo:hi] > 4):
                i_rim = i - n_off + 1; break
    if i_rim is None:
        return None

    # Compute candidate direction signals
    eps = 1e-9
    fz_for = np.where(np.abs(fz_b_smooth) < 0.5, np.copysign(0.5, fz_b_smooth + eps), fz_b_smooth)
    rcop_x = -ty_b / fz_for
    rcop_y =  tx_b / fz_for
    rcop_mag = np.hypot(rcop_x, rcop_y)
    F_lat_mag = np.hypot(fx_b, fy_b)

    # Direction signals (each unit-normalized)
    def unit(x, y):
        n_ = np.maximum(np.hypot(x, y), eps)
        return x / n_, y / n_

    s1_x, s1_y = unit(-rcop_x, -rcop_y)              # -r_cop
    s2_x, s2_y = unit(-fx_b, -fy_b)                  # -F_lat_sensed
    s3_x, s3_y = unit( fx_b,  fy_b)                  # +F_lat_sensed
    s4_x, s4_y = unit(vx, vy)                        # peg velocity (drag)

    # Targets
    rcx = tcp_x[i_rim]; rcy = tcp_y[i_rim]
    cx0 = tcp_x[i_contact]; cy0 = tcp_y[i_contact]
    # T2: contact→rim_cross overall direction (constant per demo)
    dt2x, dt2y = unit(np.array(rcx - cx0), np.array(rcy - cy0))
    dt2x = float(dt2x); dt2y = float(dt2y)
    # T3: per-tick (rim_cross - current_xy) direction (dynamic target)
    t3x_arr, t3y_arr = unit(rcx - tcp_x, rcy - tcp_y)

    # Filter mask: search segment + high fz + moving
    mask_search = np.zeros(n, dtype=bool)
    mask_search[i_contact:i_rim] = True
    mask = mask_search & (abs_fz_b > 4) & (spd > 5e-4)

    if mask.sum() < 10:
        return None

    def align(sx, sy, tx, ty):
        return sx * tx + sy * ty

    # Time-resolved alignment with target T2 (overall hole direction, constant)
    # Quartile breakdown of search segment to see temporal evolution
    search_indices = np.arange(i_contact, i_rim)
    q1_end = i_contact + (i_rim - i_contact) // 4
    q4_start = i_contact + 3 * (i_rim - i_contact) // 4
    mask_q1 = mask.copy(); mask_q1[q1_end:] = False
    mask_q4 = mask.copy(); mask_q4[:q4_start] = False

    out = {"basename": bn,
           "i_contact": int(i_contact), "i_rim": int(i_rim),
           "search_dur_s": float(ts[i_rim] - ts[i_contact]),
           "n_filtered": int(mask.sum()),
           "n_q1": int(mask_q1.sum()), "n_q4": int(mask_q4.sum()),
           "F_lat_mean_search_N": float(np.mean(F_lat_mag[mask])),
           "F_lat_q4_mean_N": float(np.mean(F_lat_mag[mask_q4])) if mask_q4.sum() else float("nan"),
           "rcop_mag_mean_search_mm": float(1000 * np.mean(rcop_mag[mask])),
           "rcop_mag_q4_mean_mm": float(1000 * np.mean(rcop_mag[mask_q4])) if mask_q4.sum() else float("nan"),
           "abs_fz_search_mean_N": float(np.mean(abs_fz_b[mask])),
           }

    for sname, sx, sy in [("S1_neg_rcop", s1_x, s1_y),
                           ("S2_neg_Flat", s2_x, s2_y),
                           ("S3_pos_Flat", s3_x, s3_y),
                           ("S4_drag",     s4_x, s4_y)]:
        # vs T1 (instantaneous drag)
        a_t1 = sx*s4_x + sy*s4_y
        # vs T2 (overall hole direction)
        a_t2 = sx*dt2x + sy*dt2y
        # vs T3 (per-tick remaining direction to rim_cross)
        a_t3 = sx*t3x_arr + sy*t3y_arr
        out[f"{sname}_vs_T1_drag_mean"] = float(np.mean(a_t1[mask]))
        out[f"{sname}_vs_T2_overall_mean"] = float(np.mean(a_t2[mask]))
        out[f"{sname}_vs_T3_remaining_mean"] = float(np.mean(a_t3[mask]))
        out[f"{sname}_vs_T2_q1_mean"] = float(np.mean(a_t2[mask_q1])) if mask_q1.sum() else float("nan")
        out[f"{sname}_vs_T2_q4_mean"] = float(np.mean(a_t2[mask_q4])) if mask_q4.sum() else float("nan")

    return out


def main():
    rows = []
    print(f"{'demo':<15s} {'dur':>5s} {'|F_lat|':>8s} {'|rcop|':>8s} | "
          f"{'S1 vs T2':>8s} {'S2 vs T2':>8s} {'S3 vs T2':>8s} {'S4 vs T2':>8s} | "
          f"{'S1 q1→q4':>10s} {'S2 q1→q4':>10s}")
    for bn in CLEAN_DEMOS:
        r = _process(bn)
        if r is None: continue
        rows.append(r)
        print(f"{bn[-15:]:<15s} {r['search_dur_s']:>5.2f} "
              f"{r['F_lat_mean_search_N']:>8.2f} {r['rcop_mag_mean_search_mm']:>8.1f} | "
              f"{r['S1_neg_rcop_vs_T2_overall_mean']:>+8.2f} "
              f"{r['S2_neg_Flat_vs_T2_overall_mean']:>+8.2f} "
              f"{r['S3_pos_Flat_vs_T2_overall_mean']:>+8.2f} "
              f"{r['S4_drag_vs_T2_overall_mean']:>+8.2f} | "
              f"{r['S1_neg_rcop_vs_T2_q1_mean']:>+5.2f}→{r['S1_neg_rcop_vs_T2_q4_mean']:>+5.2f}  "
              f"{r['S2_neg_Flat_vs_T2_q1_mean']:>+5.2f}→{r['S2_neg_Flat_vs_T2_q4_mean']:>+5.2f}")

    # Cross-demo aggregates
    def agg(key):
        return np.array([r[key] for r in rows])
    print()
    print("=== Cross-demo means (alignment with overall direction-to-hole T2) ===")
    print(f"  S1 (-r_cop):          mean={np.mean(agg('S1_neg_rcop_vs_T2_overall_mean')):+.3f}")
    print(f"  S2 (-F_lat_sensed):   mean={np.mean(agg('S2_neg_Flat_vs_T2_overall_mean')):+.3f}")
    print(f"  S3 (+F_lat_sensed):   mean={np.mean(agg('S3_pos_Flat_vs_T2_overall_mean')):+.3f}")
    print(f"  S4 (drag direction):  mean={np.mean(agg('S4_drag_vs_T2_overall_mean')):+.3f}")
    print()
    print("=== Cross-demo means: alignment with INSTANTANEOUS drag direction T1 ===")
    print(f"  S1 (-r_cop) vs drag:        {np.mean(agg('S1_neg_rcop_vs_T1_drag_mean')):+.3f}")
    print(f"  S2 (-F_lat) vs drag:        {np.mean(agg('S2_neg_Flat_vs_T1_drag_mean')):+.3f}")
    print(f"  S3 (+F_lat) vs drag:        {np.mean(agg('S3_pos_Flat_vs_T1_drag_mean')):+.3f}")
    print()
    print("=== Cross-demo means: alignment with REMAINING-distance-to-rim-cross T3 ===")
    print(f"  S1 (-r_cop) vs remaining:   {np.mean(agg('S1_neg_rcop_vs_T3_remaining_mean')):+.3f}")
    print(f"  S2 (-F_lat) vs remaining:   {np.mean(agg('S2_neg_Flat_vs_T3_remaining_mean')):+.3f}")
    print(f"  S3 (+F_lat) vs remaining:   {np.mean(agg('S3_pos_Flat_vs_T3_remaining_mean')):+.3f}")
    print()
    print("=== Temporal evolution Q1 → Q4 (last quarter of search) ===")
    print(f"  S1 (-r_cop): {np.mean(agg('S1_neg_rcop_vs_T2_q1_mean')):+.3f} → {np.mean(agg('S1_neg_rcop_vs_T2_q4_mean')):+.3f}")
    print(f"  S2 (-F_lat): {np.mean(agg('S2_neg_Flat_vs_T2_q1_mean')):+.3f} → {np.mean(agg('S2_neg_Flat_vs_T2_q4_mean')):+.3f}")
    print()
    print("=== |F_lat| and |r_cop| evolution mean → q4 ===")
    flmean = np.array([r['F_lat_mean_search_N'] for r in rows])
    flq4 = np.array([r['F_lat_q4_mean_N'] for r in rows])
    rmmean = np.array([r['rcop_mag_mean_search_mm'] for r in rows])
    rmq4 = np.array([r['rcop_mag_q4_mean_mm'] for r in rows])
    print(f"  |F_lat| (N):  search={np.mean(flmean):.2f}  q4(near rim-cross)={np.mean(flq4):.2f}")
    print(f"  |r_cop| (mm): search={np.mean(rmmean):.1f}  q4(near rim-cross)={np.mean(rmq4):.1f}")

    out = {"per_demo": rows}
    (DATA_DIR / "search_lat_force_analysis.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
