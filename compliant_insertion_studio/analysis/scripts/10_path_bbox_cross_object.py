# Reference: extends scripts/05_fail_motion_pattern.py to compute post-contact 10s
# path-length, bbox-diagonal, and path/bbox ratio across all 4 GOLD object groups.
# Tests cross-object portability of I006 (path/bbox = wide-arc vs tight-spiral discriminator).
import json, os, sys, math
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from _paths import DATA_DIR

OUT_DIR = str(DATA_DIR)
SUMMARIES = json.load(open(os.path.join(OUT_DIR, "summaries.json")))


def episode_metrics(r, window_s=10.0):
    ps_path = os.path.join(OUT_DIR, "per_sample",
                           os.path.basename(r["csv_path"]).replace(".csv", ".per_sample.json"))
    if not os.path.exists(ps_path):
        return None
    ps = json.load(open(ps_path))
    ci = r["contact_idx_active"]
    if ci is None:
        return None
    t = np.array(ps["t_s"], dtype=float)
    tcp_x = np.array(ps["tcp_x"], dtype=float)
    tcp_y = np.array(ps["tcp_y"], dtype=float)
    n = len(t)
    if ci >= n - 5:
        return None
    t_end_target = t[ci] + window_s
    i_end = n
    for i in range(ci, n):
        if t[i] >= t_end_target:
            i_end = i
            break
    xs = tcp_x[ci:i_end]
    ys = tcp_y[ci:i_end]
    mask = np.isfinite(xs) & np.isfinite(ys)
    xs = xs[mask]
    ys = ys[mask]
    if len(xs) < 5:
        return None
    dxs = np.diff(xs)
    dys = np.diff(ys)
    path_len_mm = float(np.sum(np.sqrt(dxs * dxs + dys * dys))) * 1000.0
    bbox_diag_mm = math.hypot((xs.max() - xs.min()) * 1000.0, (ys.max() - ys.min()) * 1000.0)
    if bbox_diag_mm < 0.1:
        ratio = float("nan")
    else:
        ratio = path_len_mm / bbox_diag_mm
    return {
        "path_len_mm": path_len_mm,
        "bbox_diag_mm": bbox_diag_mm,
        "path_bbox_ratio": ratio,
        "duration_s": float(t[i_end - 1] - t[ci]),
        "final_z_drop_mm": r.get("final_z_drop_mm"),
    }


def pq(arr):
    a = np.array([x for x in arr if np.isfinite(x)])
    if a.size == 0:
        return float("nan"), float("nan"), float("nan"), 0
    return (
        float(np.percentile(a, 25)),
        float(np.percentile(a, 50)),
        float(np.percentile(a, 75)),
        int(a.size),
    )


def main():
    groups = [
        ("GOLD u_orange",        lambda r: r["object"] == "u_orange"           and r["outcome"] == "success" and "20260503" in r["csv_path"]),
        ("GOLD u_brown",         lambda r: r["object"] == "u_brown"            and r["outcome"] == "success" and "20260503" in r["csv_path"]),
        ("GOLD inv_u_yellow",    lambda r: r["object"] == "inverted_u_yellow"  and r["outcome"] == "success" and "20260503" in r["csv_path"]),
        ("GOLD line_green",      lambda r: r["object"] == "line_green"         and r["outcome"] == "success" and "20260503" in r["csv_path"]),
        ("FAIL u_orange",        lambda r: r["object"] == "u_orange"           and r["outcome"] == "abort"   and "20260504" in r["csv_path"] and r.get("final_z_drop_mm", 99) < 5.0),
    ]
    out = {"window_s": 10.0, "groups": {}}
    print(f"{'group':<22s} {'n':>3s} {'path_len_mm (p25/p50/p75)':>32s} {'bbox_diag_mm (p25/p50/p75)':>32s} {'ratio (p25/p50/p75)':>28s}")
    print("-" * 120)
    for label, pred in groups:
        plens, bbs, rats = [], [], []
        per_ep = []
        for r in SUMMARIES:
            if not pred(r):
                continue
            m = episode_metrics(r)
            if m is None:
                continue
            plens.append(m["path_len_mm"])
            bbs.append(m["bbox_diag_mm"])
            rats.append(m["path_bbox_ratio"])
            per_ep.append({"csv": os.path.basename(r["csv_path"]), **m})
        p25_pl, p50_pl, p75_pl, n_pl = pq(plens)
        p25_bb, p50_bb, p75_bb, _    = pq(bbs)
        p25_rt, p50_rt, p75_rt, _    = pq(rats)
        out["groups"][label] = {
            "n": n_pl,
            "path_len_mm":   {"p25": p25_pl, "p50": p50_pl, "p75": p75_pl},
            "bbox_diag_mm":  {"p25": p25_bb, "p50": p50_bb, "p75": p75_bb},
            "path_bbox_ratio": {"p25": p25_rt, "p50": p50_rt, "p75": p75_rt},
            "per_episode": per_ep,
        }
        print(f"{label:<22s} {n_pl:>3d} "
              f"{p25_pl:>+8.2f}/{p50_pl:>+6.2f}/{p75_pl:>+6.2f}     "
              f"{p25_bb:>+5.2f}/{p50_bb:>+5.2f}/{p75_bb:>+5.2f}     "
              f"{p25_rt:>+5.2f}/{p50_rt:>+5.2f}/{p75_rt:>+5.2f}")

    # Cross-object portability summary
    gold_keys = ["GOLD u_orange", "GOLD u_brown", "GOLD inv_u_yellow", "GOLD line_green"]
    gold_ratios = [out["groups"][k]["path_bbox_ratio"]["p50"] for k in gold_keys]
    gold_bbs    = [out["groups"][k]["bbox_diag_mm"]["p50"]    for k in gold_keys]
    fail_ratio  = out["groups"]["FAIL u_orange"]["path_bbox_ratio"]["p50"]
    fail_bbox   = out["groups"]["FAIL u_orange"]["bbox_diag_mm"]["p50"]

    gold_ratio_min = float(np.nanmin(gold_ratios))
    gold_ratio_max = float(np.nanmax(gold_ratios))
    gold_ratio_spread = gold_ratio_max - gold_ratio_min

    print()
    print("=== Cross-object portability summary ===")
    print(f"GOLD path/bbox ratios per object  : {dict(zip(gold_keys, [round(x, 2) for x in gold_ratios]))}")
    print(f"GOLD ratio spread (max-min)       : {gold_ratio_spread:.2f}")
    print(f"GOLD bbox medians per object (mm) : {dict(zip(gold_keys, [round(x, 2) for x in gold_bbs]))}")
    print(f"FAIL u_orange ratio (median)      : {fail_ratio:.2f}")
    print(f"FAIL u_orange bbox  (median, mm)  : {fail_bbox:.2f}")
    # Decision rule for portability
    threshold = (gold_ratio_max + fail_ratio) / 2.0
    portable_threshold = (fail_ratio - gold_ratio_max) > 0.5  # at least 0.5-unit gap
    print(f"Suggested universal threshold     : {threshold:.2f}  (separates worst GOLD from FAIL u_orange)")
    print(f"GOLD/FAIL gap > 0.5                : {portable_threshold}")

    out["portability"] = {
        "gold_ratio_min": gold_ratio_min,
        "gold_ratio_max": gold_ratio_max,
        "gold_ratio_spread": gold_ratio_spread,
        "fail_u_orange_ratio": fail_ratio,
        "suggested_threshold_ratio": threshold,
        "gold_fail_gap": fail_ratio - gold_ratio_max,
        "all_gold_below_fail": all(g < fail_ratio for g in gold_ratios if np.isfinite(g)),
    }

    out_path = os.path.join(OUT_DIR, "path_bbox_cross_object.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
