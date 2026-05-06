"""Discovery 006: Test H110 — do per-attempt post-contact feature aggregates
discriminate u_orange AUTO_success from AUTO_fail?

Discovery 004 (I012): no per-timepoint median trajectory leaves the success band.
Discovery 005 (I013): at-contact xy offset does not discriminate (AUC ~0.57).
By elimination, the signal must lie in either (a) per-attempt aggregates that
wash out in the per-timepoint median, or (b) signals not currently recorded
(G001-G004). This iteration tests (a).

Method:
  - Subset: u_orange autonomous, session 20260504.
  - Class: AUTO_success final_z_drop_mm >= 20; AUTO_fail < 5.
  - Window: [contact_t_s, contact_t_s + 5.0] in per_sample.json t_s coordinates.
  - For each of 15 candidate aggregates, compute single-threshold AUC in the
    direction (high or low → success) that maximizes AUC; Mann-Whitney p; medians.
  - Rank features by AUC.
  - Anti-leakage: do NOT include final_z_drop or any aggregate that monotonically
    reflects the label (e.g., terminal z_drop). dz_dt aggregates ARE allowed
    because dz_dt at t<+5s is dynamic, not a label.

Usage:
    python3 15_h110_post_contact_aggregates.py [out_path.json]
"""
import json
import sys
from math import erf, sqrt
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import DATA_DIR, PER_SAMPLE_DIR  # noqa: E402

SUMMARIES = json.load(open(DATA_DIR / "summaries.json"))

OBJECT = "u_orange"
SESSION_TAG = "20260504"
SUCCESS_THRESH_MM = 20.0
FAIL_THRESH_MM = 5.0
WINDOW_S = 5.0


def classify(s):
    name = Path(s["csv_path"]).name
    if s["object"] != OBJECT or SESSION_TAG not in name:
        return None
    fdz = s.get("final_z_drop_mm") or 0.0
    if fdz >= SUCCESS_THRESH_MM:
        return "AUTO_success"
    if fdz < FAIL_THRESH_MM:
        return "AUTO_fail"
    return None


def per_sample_for(s):
    p = PER_SAMPLE_DIR / (Path(s["csv_path"]).stem + ".per_sample.json")
    if not p.exists():
        return None
    return json.load(open(p))


def slice_window(ps, contact_t_s, window_s=WINDOW_S):
    """Return dict of arrays sliced to [contact_t_s, contact_t_s + window_s].
    Returns None if fewer than 10 samples in the window."""
    t = np.asarray(ps.get("t_s", []), dtype=float)
    if len(t) == 0:
        return None
    mask = (t >= contact_t_s) & (t <= contact_t_s + window_s)
    if mask.sum() < 10:
        return None
    out = {"t": t[mask]}
    for k in ("F_lat_base", "F_lat_dir_base_rad", "F_lat_tool",
              "fz_t", "fx_t", "fy_t", "tx_t", "ty_t",
              "tcp_x", "tcp_y", "tcp_z",
              "tilt_deg", "r_cop_m", "xy_excursion_mm",
              "z_drop_mm", "dz_dt_mm_s"):
        v = ps.get(k)
        if v is None:
            out[k] = None
        else:
            arr = np.asarray(v, dtype=float)
            if len(arr) != len(t):
                out[k] = None
            else:
                out[k] = arr[mask]
    return out


def max_run_length_above(values, t, thresh):
    """Maximum sustained duration (s) for which values > thresh."""
    if values is None or len(values) == 0:
        return 0.0
    above = values > thresh
    best = 0.0
    cur_start = None
    for i, hi in enumerate(above):
        if hi and cur_start is None:
            cur_start = t[i]
        elif not hi and cur_start is not None:
            best = max(best, float(t[i] - cur_start))
            cur_start = None
    if cur_start is not None:
        best = max(best, float(t[-1] - cur_start))
    return best


def trapz_safe(y, x):
    if y is None or len(y) < 2:
        return 0.0
    return float(np.trapz(np.abs(y), x))


def circular_var(angles_rad):
    """Mardia-Jupp circular variance: 1 - |mean(e^{i theta})|. Range [0, 1]."""
    if angles_rad is None or len(angles_rad) == 0:
        return float("nan")
    valid = np.isfinite(angles_rad)
    if valid.sum() == 0:
        return float("nan")
    a = angles_rad[valid]
    return float(1.0 - np.abs(np.mean(np.exp(1j * a))))


def aggregates(win):
    """Compute every candidate aggregate over the windowed dict. Returns
    {name: float}. NaN where input array is missing."""
    t = win["t"]
    F = win["F_lat_base"]
    fz = win["fz_t"]
    tilt = win["tilt_deg"]
    rcop = win["r_cop_m"]
    xy_exc = win["xy_excursion_mm"]
    dz = win["dz_dt_mm_s"]
    F_dir = win["F_lat_dir_base_rad"]

    out = {}
    out["max_F_lat_base_N"] = float(np.nanmax(F)) if F is not None else float("nan")
    out["mean_F_lat_base_N"] = float(np.nanmean(F)) if F is not None else float("nan")
    out["time_F_lat_base_above_2N_s"] = max_run_length_above(F, t, 2.0) if F is not None else float("nan")
    out["time_F_lat_base_above_3N_s"] = max_run_length_above(F, t, 3.0) if F is not None else float("nan")
    out["integral_xy_excursion_mm_s"] = trapz_safe(xy_exc, t) if xy_exc is not None else float("nan")
    out["max_xy_excursion_mm"] = float(np.nanmax(xy_exc)) if xy_exc is not None else float("nan")
    out["max_tilt_deg"] = float(np.nanmax(tilt)) if tilt is not None else float("nan")
    out["time_tilt_above_1deg_s"] = max_run_length_above(tilt, t, 1.0) if tilt is not None else float("nan")
    out["max_r_cop_m"] = float(np.nanmax(rcop)) if rcop is not None else float("nan")
    out["mean_r_cop_m"] = float(np.nanmean(rcop)) if rcop is not None else float("nan")
    out["max_abs_fz_t_N"] = float(np.nanmax(np.abs(fz))) if fz is not None else float("nan")
    out["min_abs_fz_t_N"] = float(np.nanmin(np.abs(fz))) if fz is not None else float("nan")
    out["mean_dz_dt_mm_s"] = float(np.nanmean(dz)) if dz is not None else float("nan")
    out["frac_descending"] = float(np.mean(dz < 0)) if dz is not None else float("nan")
    out["F_lat_dir_circular_var"] = circular_var(F_dir)
    return out


def auc_one_sided(succ_vals, fail_vals):
    """Returns (auc_high_predicts_success, auc_low_predicts_success). Skips NaN."""
    s = np.asarray([v for v in succ_vals if np.isfinite(v)], dtype=float)
    f = np.asarray([v for v in fail_vals if np.isfinite(v)], dtype=float)
    if len(s) == 0 or len(f) == 0:
        return None, None
    wins_high = 0
    wins_low = 0
    ties = 0
    for x in s:
        wins_high += np.sum(f < x)
        wins_low += np.sum(f > x)
        ties += np.sum(f == x)
    n = len(s) * len(f)
    return (
        float((wins_high + 0.5 * ties) / n),
        float((wins_low + 0.5 * ties) / n),
    )


def mannwhitney_one_sided(a, b, alt):
    """One-sided rank-sum normal approx with tie correction in mean ranks.
    alt: 'a_lt_b' or 'a_gt_b'."""
    a = np.asarray([v for v in a if np.isfinite(v)], dtype=float)
    b = np.asarray([v for v in b if np.isfinite(v)], dtype=float)
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return None, None
    combined = np.concatenate([a, b])
    order = np.argsort(combined, kind="mergesort")
    ranks = np.empty(len(combined))
    ranks[order] = np.arange(1, len(combined) + 1)
    sorted_vals = combined[order]
    i = 0
    while i < len(sorted_vals):
        j = i
        while j + 1 < len(sorted_vals) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        if j > i:
            avg = ranks[order[i:j + 1]].mean()
            ranks[order[i:j + 1]] = avg
        i = j + 1
    R1 = ranks[:n1].sum()
    U1 = R1 - n1 * (n1 + 1) / 2
    mu = n1 * n2 / 2
    sigma = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    z = (U1 - mu) / sigma if sigma > 0 else 0.0
    p_lt = 0.5 * (1 + erf(z / sqrt(2)))
    p_gt = 1 - p_lt
    return float(U1), float(p_lt if alt == "a_lt_b" else p_gt)


def main():
    succ_features = []
    fail_features = []
    skipped = {"no_per_sample": 0, "no_window": 0, "wrong_class": 0,
               "no_contact_t_s": 0}

    for r in SUMMARIES:
        cls = classify(r)
        if cls is None:
            skipped["wrong_class"] += 1
            continue
        ct = r.get("contact_t_s")
        if ct is None or not np.isfinite(ct):
            skipped["no_contact_t_s"] += 1
            continue
        ps = per_sample_for(r)
        if ps is None:
            skipped["no_per_sample"] += 1
            continue
        win = slice_window(ps, float(ct))
        if win is None:
            skipped["no_window"] += 1
            continue
        ag = aggregates(win)
        ag["_csv"] = Path(r["csv_path"]).name
        if cls == "AUTO_success":
            succ_features.append(ag)
        else:
            fail_features.append(ag)

    print(f"AUTO_success n={len(succ_features)}  AUTO_fail n={len(fail_features)}")
    print(f"skipped={skipped}")
    if len(succ_features) < 5 or len(fail_features) < 5:
        print("ERROR: insufficient samples")
        sys.exit(1)

    feat_names = [k for k in succ_features[0].keys() if not k.startswith("_")]

    rows = []
    for fname in feat_names:
        s_vals = [r[fname] for r in succ_features]
        f_vals = [r[fname] for r in fail_features]
        s_arr = np.asarray([v for v in s_vals if np.isfinite(v)], dtype=float)
        f_arr = np.asarray([v for v in f_vals if np.isfinite(v)], dtype=float)
        if len(s_arr) < 5 or len(f_arr) < 5:
            continue
        auc_hi, auc_lo = auc_one_sided(s_vals, f_vals)
        if auc_hi >= auc_lo:
            direction = "high_predicts_success"
            auc = auc_hi
            U, p = mannwhitney_one_sided(s_vals, f_vals, "a_gt_b")
        else:
            direction = "low_predicts_success"
            auc = auc_lo
            U, p = mannwhitney_one_sided(s_vals, f_vals, "a_lt_b")

        rows.append({
            "feature": fname,
            "n_succ": int(len(s_arr)),
            "n_fail": int(len(f_arr)),
            "succ_median": float(np.median(s_arr)),
            "succ_p25": float(np.percentile(s_arr, 25)),
            "succ_p75": float(np.percentile(s_arr, 75)),
            "fail_median": float(np.median(f_arr)),
            "fail_p25": float(np.percentile(f_arr, 25)),
            "fail_p75": float(np.percentile(f_arr, 75)),
            "delta_median": float(np.median(s_arr) - np.median(f_arr)),
            "direction": direction,
            "auc": auc,
            "mannwhitney_U": U,
            "mannwhitney_p_one_sided": p,
        })

    rows.sort(key=lambda r: r["auc"], reverse=True)

    best = rows[0] if rows else None
    interpretation = ""
    if best is None:
        interpretation = "no_features"
    elif best["auc"] >= 0.75:
        interpretation = (
            f"STRONG: best single-feature AUC = {best['auc']:.3f} ({best['feature']}, "
            f"{best['direction']}). Promote H110_{best['feature']} to invariant; "
            f"propose FSM patch."
        )
    elif best["auc"] >= 0.65:
        interpretation = (
            f"MODERATE: best AUC = {best['auc']:.3f} ({best['feature']}). "
            f"Worth pursuing as candidate; not yet a strong discriminator."
        )
    else:
        interpretation = (
            f"NO_SEPARATION: best single-feature AUC = {best['auc']:.3f} "
            f"({best['feature']}, {best['direction']}). "
            f"Combined with I012 (no median band leaving) and I013 (no at-contact "
            f"xy separation), this confirms the elimination chain: the success/fail "
            f"signal is NOT in any currently-recorded post-contact aggregate. "
            f"Schema bump G001-G004 (joint_states / native-rate wrench / cmd_fxy) "
            f"is now the only remaining source of new signal."
        )

    out = {
        "object": OBJECT,
        "session_tag": SESSION_TAG,
        "window_s": WINDOW_S,
        "n_AUTO_success": len(succ_features),
        "n_AUTO_fail": len(fail_features),
        "skipped": skipped,
        "thresholds": {
            "auc_strong": 0.75,
            "auc_moderate": 0.65,
            "auc_no_separation": 0.60,
        },
        "feature_ranking_by_auc": rows,
        "best_feature": best,
        "interpretation": interpretation,
    }

    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("metrics.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}\n")

    print("--- H110 feature ranking by AUC (descending) ---\n")
    print(f"{'feature':<32} {'AUC':>6} {'dir':<22} {'p1side':>8} "
          f"{'succ_med':>10} {'fail_med':>10} {'Δmed':>9}")
    for r in rows:
        p_str = f"{r['mannwhitney_p_one_sided']:.4f}" if r['mannwhitney_p_one_sided'] is not None else "—"
        print(f"{r['feature']:<32} {r['auc']:>6.3f} {r['direction']:<22} "
              f"{p_str:>8} {r['succ_median']:>10.3f} {r['fail_median']:>10.3f} "
              f"{r['delta_median']:>+9.3f}")
    print(f"\n--- INTERPRETATION ---\n{interpretation}")


if __name__ == "__main__":
    main()
