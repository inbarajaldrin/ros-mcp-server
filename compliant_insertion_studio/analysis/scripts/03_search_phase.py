# Reference: examine what happens BEFORE Fz collapse — the "search" phase the operator drives.
# For each gold demo, find the moment of Fz_t collapse (Fz_t crosses below 2N from above) and look back.
# Compute: time-to-collapse, xy excursion before collapse, F_lat (base) magnitude + direction during search.
import json, os, sys, math
import numpy as np
from collections import defaultdict
sys.path.insert(0, os.path.dirname(__file__))
from _paths import DATA_DIR

OUT_DIR = str(DATA_DIR)
SUMMARIES = json.load(open(os.path.join(OUT_DIR, "summaries.json")))

GROUPS = {
    "u_orange_GOLD":     lambda r: r["object"] == "u_orange"   and r["outcome"] == "success" and "20260503" in r["csv_path"],
    "u_brown_GOLD":      lambda r: r["object"] == "u_brown"    and r["outcome"] == "success" and "20260503" in r["csv_path"],
    "inv_u_yellow_GOLD": lambda r: r["object"] == "inverted_u_yellow" and r["outcome"] == "success" and "20260503" in r["csv_path"],
    "line_green_GOLD":   lambda r: r["object"] == "line_green" and r["outcome"] == "success" and "20260503" in r["csv_path"],
    "u_orange_FAIL":     lambda r: r["object"] == "u_orange"   and r["outcome"] == "abort"   and "20260504" in r["csv_path"] and r["final_z_drop_mm"] < 5.0,
}

def find_fz_collapse(fz_smooth, contact_idx):
    """First post-contact index where smoothed Fz_t drops below 2 N for sustained 200ms."""
    th = 2.0
    sustain = 20  # 200ms at 100Hz
    n = len(fz_smooth)
    run = 0
    for i in range(contact_idx, n):
        v = fz_smooth[i]
        if v != v: continue
        if abs(v) < th:
            run += 1
            if run >= sustain:
                return i - run + 1
        else:
            run = 0
    return None

def smooth(arr, w=50):
    n = len(arr)
    out = [None]*n
    cs = 0.0; cn = 0
    buf = []
    for i, v in enumerate(arr):
        if v is not None and v == v:
            buf.append(v)
            if len(buf) > w: buf.pop(0)
            out[i] = sum(buf)/len(buf) if buf else None
    return out

def collect():
    by_group = defaultdict(list)
    for r in SUMMARIES:
        for gn, p in GROUPS.items():
            if p(r):
                by_group[gn].append(r)
    return by_group

def angle_circular_stats(angles_rad, weights):
    """Weighted circular mean + concentration kappa-ish (resultant length)."""
    if len(angles_rad) == 0: return None, None
    w = np.array(weights)
    a = np.array(angles_rad)
    mask = (w > 0) & np.isfinite(a)
    if mask.sum() == 0: return None, None
    w = w[mask]; a = a[mask]
    sx = np.sum(w * np.cos(a)); sy = np.sum(w * np.sin(a))
    mean = math.atan2(sy, sx)
    R_ = math.sqrt(sx*sx + sy*sy) / np.sum(w)
    return math.degrees(mean), R_  # mean in deg [-180,180], concentration in [0,1]

def main():
    by_group = collect()
    print(f"\n{'group':22s} n   t_to_collapse(s)  xy_excur_pre(mm)  F_lat_pre(N)  F_lat_dir_mean(deg)  R(0-1)  Fz_t_pre(N)  cmd_fz_pre(N)")
    rows_export = {}
    for gn, rs in by_group.items():
        per_episode = []
        all_dirs = []
        all_weights = []
        for r in rs:
            ps_path = os.path.join(OUT_DIR, "per_sample",
                                   os.path.basename(r["csv_path"]).replace(".csv", ".per_sample.json"))
            if not os.path.exists(ps_path): continue
            ps = json.load(open(ps_path))
            ci = r["contact_idx_active"]
            t = ps["t_s"]
            fz_t = ps["fz_t"]
            fz_t_s = smooth(fz_t, 50)
            collapse_idx = find_fz_collapse(fz_t_s, ci)
            if collapse_idx is None:
                continue
            t_collapse = t[collapse_idx] - t[ci]  # seconds
            # pre-collapse window (last 1.0s before collapse, or from contact if shorter)
            tc = t[collapse_idx]
            t0 = max(t[ci], tc - 1.0)
            # find slice
            i0 = ci
            for i in range(ci, collapse_idx):
                if t[i] >= t0:
                    i0 = i; break
            # xy excursion in pre-collapse window
            xs = ps["tcp_x"][i0:collapse_idx]
            ys = ps["tcp_y"][i0:collapse_idx]
            xy_excur = (max(xs)-min(xs))*1000, (max(ys)-min(ys))*1000
            xy_excur_mm = math.hypot(xy_excur[0], xy_excur[1])
            # Mean F_lat in base frame (signed direction!)
            fxb = ps["fx_b"][i0:collapse_idx]; fyb = ps["fy_b"][i0:collapse_idx]
            f_lat_mag = [math.hypot(a,b) for a,b in zip(fxb,fyb) if a==a and b==b]
            f_lat_dir = [math.atan2(b,a) for a,b in zip(fxb,fyb) if a==a and b==b]
            f_lat_avg = sum(f_lat_mag)/len(f_lat_mag) if f_lat_mag else None
            # Pre-collapse Fz_t mean
            fzs = [v for v in fz_t[i0:collapse_idx] if v==v]
            fz_avg = sum(fzs)/len(fzs) if fzs else None
            # Commanded fz pre-collapse
            cfz = [v for v in ps["commanded_fz"][i0:collapse_idx] if v==v]
            cfz_avg = sum(cfz)/len(cfz) if cfz else None

            per_episode.append({
                "csv": os.path.basename(r["csv_path"]),
                "t_to_collapse_s": t_collapse,
                "xy_excur_pre_mm": xy_excur_mm,
                "F_lat_pre_N": f_lat_avg,
                "Fz_t_pre_N": fz_avg,
                "cmd_fz_pre_N": cfz_avg,
                "f_lat_dirs_rad": f_lat_dir,
                "f_lat_mags_N": f_lat_mag,
            })
            all_dirs.extend(f_lat_dir)
            all_weights.extend(f_lat_mag)

        if not per_episode:
            print(f"{gn:22s} -- no collapse detected --")
            continue
        n = len(per_episode)
        med = lambda key: float(np.median([p[key] for p in per_episode if p[key] is not None]))
        m_t = med("t_to_collapse_s")
        m_xy = med("xy_excur_pre_mm")
        m_fl = med("F_lat_pre_N")
        m_fz = med("Fz_t_pre_N")
        m_cfz = med("cmd_fz_pre_N")
        mean_dir, R_ = angle_circular_stats(all_dirs, all_weights)
        print(f"{gn:22s} {n:>2d}   {m_t:>+8.2f}        {m_xy:>+6.2f}       {m_fl:>+5.2f}     "
              f"{(mean_dir if mean_dir is not None else float('nan')):>+7.1f}             "
              f"{(R_ if R_ is not None else float('nan')):>4.2f}    {m_fz:>+5.2f}        {m_cfz:>+5.2f}")
        rows_export[gn] = per_episode
    json.dump(rows_export, open(os.path.join(OUT_DIR, "search_phase.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
