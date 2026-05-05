"""Discovery 005: Test H109 — is initial xy offset (contact_xy − seat_xy) the
discriminator between AUTO_success and AUTO_fail u_orange attempts?

Discovery 004 showed no per-timepoint median post-contact feature discriminates
the two classes. Interpretation (A): the discriminator is at-contact — lucky
landings near the chamfer drop in; unlucky landings on the rim cannot.

Method:
  - Load summaries.json (already has contact_xy_m + final_z_drop_mm per attempt).
  - For AUTO_success (final_z_drop_mm >= 20): compute actual_seat_xy from the
    last ACTIVE sample of the per_sample.json (tcp_x, tcp_y at min depth).
  - Reference seat_xy = MEDIAN of AUTO_success actual_seat_xy values (data-driven,
    workspace-invariant). Cross-check against STATE.json predicted_seat_xy_m.
  - Compute offset_mm = ||contact_xy - reference_seat_xy|| for every attempt.
  - Compare distributions: AUTO_success vs AUTO_fail.
  - Compute Mann–Whitney U (rank-sum) p-value as separation strength.
  - Compute single-threshold AUC (treat low offset as predictor of success).

If AUTO_success has systematically lower offset → H109 confirmed → propose I013
+ FSM patch H109.

Usage:
    python3 14_h109_initial_xy_offset.py [out_path.json]
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import DATA_DIR, PER_SAMPLE_DIR  # noqa: E402

SUMMARIES = json.load(open(DATA_DIR / "summaries.json"))

OBJECT = "u_orange"
SESSION_TAG = "20260504"
SUCCESS_THRESH_MM = 20.0
FAIL_THRESH_MM = 5.0
PREDICTED_SEAT_XY = (0.0341, -0.3635)  # STATE.json:current_target.predicted_seat_xy_m


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


def actual_seat_xy(s, ps):
    """For success attempts: xy at deepest tcp_z in the active phase."""
    tcp_x = np.asarray(ps.get("tcp_x", []), dtype=float)
    tcp_y = np.asarray(ps.get("tcp_y", []), dtype=float)
    tcp_z = np.asarray(ps.get("tcp_z", []), dtype=float)
    if not (len(tcp_x) == len(tcp_y) == len(tcp_z)) or len(tcp_z) == 0:
        return None
    valid = np.isfinite(tcp_z)
    if not valid.any():
        return None
    deepest = int(np.argmin(np.where(valid, tcp_z, np.inf)))
    return float(tcp_x[deepest]), float(tcp_y[deepest])


def mannwhitney_u(a, b):
    """One-sided rank-sum: H1 = a < b. Returns (U, p_approx_normal)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return None, None
    combined = np.concatenate([a, b])
    order = np.argsort(combined, kind="mergesort")
    ranks = np.empty(len(combined))
    ranks[order] = np.arange(1, len(combined) + 1)
    # tie correction: average ranks for equal values
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
    U2 = n1 * n2 - U1
    # Normal approximation (no tie correction in variance — fine for ~100 samples)
    mu = n1 * n2 / 2
    sigma = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    z = (U1 - mu) / sigma if sigma > 0 else 0.0
    # one-sided p (a < b means U1 small)
    from math import erf, sqrt
    p_one = 0.5 * (1 + erf(z / sqrt(2)))
    return float(U1), float(p_one)


def auc_threshold(succ_offsets, fail_offsets):
    """AUC of 'low offset → predict success'. = P(succ_offset < fail_offset)."""
    s = np.asarray(succ_offsets)
    f = np.asarray(fail_offsets)
    if len(s) == 0 or len(f) == 0:
        return None
    wins = 0
    ties = 0
    for x in s:
        wins += np.sum(f > x)
        ties += np.sum(f == x)
    return float((wins + 0.5 * ties) / (len(s) * len(f)))


def main():
    succ_records = []
    fail_records = []
    skipped = {"no_per_sample": 0, "no_contact_xy": 0, "wrong_class": 0}

    for r in SUMMARIES:
        cls = classify(r)
        if cls is None:
            skipped["wrong_class"] += 1
            continue
        cxy = r.get("contact_xy_m")
        if cxy is None or len(cxy) != 2 or not all(np.isfinite(cxy)):
            skipped["no_contact_xy"] += 1
            continue
        ps = per_sample_for(r)
        if ps is None:
            skipped["no_per_sample"] += 1
            continue
        rec = {
            "csv": Path(r["csv_path"]).name,
            "contact_xy_m": [float(cxy[0]), float(cxy[1])],
            "contact_z_m": float(r.get("contact_z_m") or 0.0),
            "final_z_drop_mm": float(r.get("final_z_drop_mm") or 0.0),
        }
        if cls == "AUTO_success":
            asxy = actual_seat_xy(r, ps)
            rec["actual_seat_xy_m"] = list(asxy) if asxy else None
            succ_records.append(rec)
        else:
            fail_records.append(rec)

    print(f"AUTO_success n={len(succ_records)}  AUTO_fail n={len(fail_records)}")
    print(f"skipped={skipped}")
    if len(succ_records) < 5 or len(fail_records) < 5:
        print("ERROR: insufficient samples")
        sys.exit(1)

    # Reference seat_xy: median of AUTO_success actual_seat_xy
    succ_seats = [r["actual_seat_xy_m"] for r in succ_records if r.get("actual_seat_xy_m")]
    succ_seats_arr = np.asarray(succ_seats)
    ref_data_seat_xy = (float(np.median(succ_seats_arr[:, 0])),
                        float(np.median(succ_seats_arr[:, 1])))
    ref_data_seat_iqr_x = float(np.percentile(succ_seats_arr[:, 0], 75) -
                                np.percentile(succ_seats_arr[:, 0], 25))
    ref_data_seat_iqr_y = float(np.percentile(succ_seats_arr[:, 1], 75) -
                                np.percentile(succ_seats_arr[:, 1], 25))

    # Compute offsets for both reference choices
    def offsets_to(ref):
        sox, soy = ref
        succ_off = [np.hypot(r["contact_xy_m"][0] - sox, r["contact_xy_m"][1] - soy) * 1000.0
                    for r in succ_records]
        fail_off = [np.hypot(r["contact_xy_m"][0] - sox, r["contact_xy_m"][1] - soy) * 1000.0
                    for r in fail_records]
        return succ_off, fail_off

    results = {}
    for label, ref in [("data_seat", ref_data_seat_xy),
                       ("predicted_seat_state_json", PREDICTED_SEAT_XY)]:
        s_off, f_off = offsets_to(ref)
        s_arr = np.asarray(s_off)
        f_arr = np.asarray(f_off)
        U, p = mannwhitney_u(s_off, f_off)
        auc = auc_threshold(s_off, f_off)
        results[label] = {
            "ref_seat_xy_m": list(ref),
            "succ_offset_mm": {
                "n": len(s_off),
                "mean": float(s_arr.mean()),
                "median": float(np.median(s_arr)),
                "p25": float(np.percentile(s_arr, 25)),
                "p75": float(np.percentile(s_arr, 75)),
                "min": float(s_arr.min()),
                "max": float(s_arr.max()),
            },
            "fail_offset_mm": {
                "n": len(f_off),
                "mean": float(f_arr.mean()),
                "median": float(np.median(f_arr)),
                "p25": float(np.percentile(f_arr, 25)),
                "p75": float(np.percentile(f_arr, 75)),
                "min": float(f_arr.min()),
                "max": float(f_arr.max()),
            },
            "delta_median_mm": float(np.median(f_arr) - np.median(s_arr)),
            "mannwhitney_U": U,
            "mannwhitney_p_one_sided_succ_lt_fail": p,
            "AUC_low_offset_predicts_success": auc,
        }

    out = {
        "object": OBJECT,
        "session_tag": SESSION_TAG,
        "n_AUTO_success": len(succ_records),
        "n_AUTO_fail": len(fail_records),
        "n_succ_with_actual_seat": len(succ_seats),
        "data_driven_reference_seat_xy_m": list(ref_data_seat_xy),
        "data_driven_reference_seat_iqr_x_m": ref_data_seat_iqr_x,
        "data_driven_reference_seat_iqr_y_m": ref_data_seat_iqr_y,
        "predicted_seat_xy_m_state_json": list(PREDICTED_SEAT_XY),
        "ref_disagreement_mm": float(np.hypot(
            ref_data_seat_xy[0] - PREDICTED_SEAT_XY[0],
            ref_data_seat_xy[1] - PREDICTED_SEAT_XY[1]) * 1000.0),
        "results": results,
        "interpretation_thresholds": {
            "AUC_>=_0.75": "strong discriminator → confirm H109",
            "AUC_>=_0.65": "moderate → H109 partially supported",
            "AUC_<_0.60": "no separation → H109 refuted",
            "p_<_0.01_AND_AUC_>=_0.65": "publish I013",
        },
    }

    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("metrics.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")

    print(f"\nData-driven reference seat (median of {len(succ_seats)} successful seat xy):")
    print(f"  ({ref_data_seat_xy[0]:+.4f}, {ref_data_seat_xy[1]:+.4f}) m")
    print(f"  IQR_x={ref_data_seat_iqr_x*1000:.2f}mm  IQR_y={ref_data_seat_iqr_y*1000:.2f}mm")
    print(f"\nState.json predicted_seat: ({PREDICTED_SEAT_XY[0]:+.4f}, {PREDICTED_SEAT_XY[1]:+.4f}) m")
    print(f"Disagreement: {out['ref_disagreement_mm']:.2f} mm")

    print("\n--- H109 test results ---\n")
    for label, d in results.items():
        print(f"[reference: {label}]")
        s = d["succ_offset_mm"]; f = d["fail_offset_mm"]
        print(f"  AUTO_success offset_mm: median={s['median']:.2f}  IQR=[{s['p25']:.2f}, {s['p75']:.2f}]  range=[{s['min']:.2f}, {s['max']:.2f}]  n={s['n']}")
        print(f"  AUTO_fail    offset_mm: median={f['median']:.2f}  IQR=[{f['p25']:.2f}, {f['p75']:.2f}]  range=[{f['min']:.2f}, {f['max']:.2f}]  n={f['n']}")
        print(f"  delta_median = {d['delta_median_mm']:+.2f} mm  (positive = fail farther from seat than success — supports H109)")
        print(f"  Mann-Whitney U = {d['mannwhitney_U']}  p_one_sided(succ<fail) = {d['mannwhitney_p_one_sided_succ_lt_fail']:.4f}")
        print(f"  AUC(low offset → success) = {d['AUC_low_offset_predicts_success']:.3f}")
        print()


if __name__ == "__main__":
    main()
