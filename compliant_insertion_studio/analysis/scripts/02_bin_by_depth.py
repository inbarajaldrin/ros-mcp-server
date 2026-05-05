# Reference: cross-episode statistics by post-contact depth band.
# Pools per-sample features across episodes within a (object, group) bucket and reports median + IQR per depth band.
import json, os, sys, glob
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from _paths import DATA_DIR

OUT_DIR = str(DATA_DIR)

# Buckets we care about.
GOLD_GROUPS = {
    # gold operator demos (May-3 only, fully seated)
    "u_orange_GOLD":          lambda r: r["object"] == "u_orange"   and r["outcome"] == "success" and "20260503" in r["csv_path"],
    "u_brown_GOLD":           lambda r: r["object"] == "u_brown"    and r["outcome"] == "success" and "20260503" in r["csv_path"],
    "inv_u_yellow_GOLD":      lambda r: r["object"] == "inverted_u_yellow" and r["outcome"] == "success" and "20260503" in r["csv_path"],
    "line_green_GOLD":        lambda r: r["object"] == "line_green" and r["outcome"] == "success" and "20260503" in r["csv_path"],
    # u_orange autonomous-failure pool (May-4 aborts that never descended)
    "u_orange_FAIL":          lambda r: r["object"] == "u_orange"   and r["outcome"] == "abort"   and "20260504" in r["csv_path"] and r["final_z_drop_mm"] < 5.0,
    # u_orange autonomous attempts that DID seat (mechanical success on autonomous, or operator near-bottom SIGTERM)
    "u_orange_AUTO_SEATED":   lambda r: r["object"] == "u_orange"   and "20260504" in r["csv_path"] and r["final_z_drop_mm"] >= 25.0,
}

# Depth bands (mm post-contact). Last band = "after seat" so we capture full descent.
DEPTH_BANDS = [(-2,0), (0,1), (1,2), (2,4), (4,7), (7,10), (10,15), (15,20), (20,25), (25,32)]

# Features to summarize
FEATURES = [
    "fz_t", "F_lat_tool", "F_lat_base", "r_cop_m",
    "fx_b", "fy_b", "fz_b",
    "tx_t", "ty_t", "tz_t",
    "dz_dt_mm_s", "tilt_deg", "xy_excursion_mm", "commanded_fz",
]


def main():
    summaries = json.load(open(os.path.join(OUT_DIR, "summaries.json")))
    # Group summaries
    grouped = defaultdict(list)
    for r in summaries:
        for gname, pred in GOLD_GROUPS.items():
            if pred(r):
                grouped[gname].append(r)

    out = {}
    for gname, rs in grouped.items():
        # bins[band_idx][feature] = list of values pooled across all episodes
        bins = [defaultdict(list) for _ in DEPTH_BANDS]
        for r in rs:
            ps_path = os.path.join(OUT_DIR, "per_sample",
                                   os.path.basename(r["csv_path"]).replace(".csv", ".per_sample.json"))
            if not os.path.exists(ps_path): continue
            ps = json.load(open(ps_path))
            ci = r["contact_idx_active"]
            if ci is None: continue
            zd = ps["z_drop_mm"][ci:]
            for f in FEATURES:
                arr = ps[f][ci:]
                for j, v in enumerate(arr):
                    if v is None: continue
                    if v != v: continue   # NaN
                    z = zd[j]
                    for bi, (lo, hi) in enumerate(DEPTH_BANDS):
                        if lo <= z < hi:
                            bins[bi][f].append(v)
                            break

        # Aggregate per band
        bands_out = []
        for bi, (lo, hi) in enumerate(DEPTH_BANDS):
            band_dict = {"depth_mm_lo": lo, "depth_mm_hi": hi, "feats": {}}
            for f in FEATURES:
                vals = bins[bi][f]
                if len(vals) < 5:
                    band_dict["feats"][f] = {"n": len(vals)}
                    continue
                a = np.array(vals)
                band_dict["feats"][f] = {
                    "n": int(len(a)),
                    "p10": float(np.percentile(a, 10)),
                    "p25": float(np.percentile(a, 25)),
                    "p50": float(np.percentile(a, 50)),
                    "p75": float(np.percentile(a, 75)),
                    "p90": float(np.percentile(a, 90)),
                    "mean": float(np.mean(a)),
                    "std": float(np.std(a)),
                }
            bands_out.append(band_dict)
        out[gname] = {"n_episodes": len(rs), "bands": bands_out}

    with open(os.path.join(OUT_DIR, "bin_stats.json"), "w") as fh:
        json.dump(out, fh, indent=2)

    # Pretty-print key features per group
    print(f"{'group':22s}  band(mm)   n_samp   fz_t.p50   F_lat_b.p50  r_cop_mm.p50  dz_dt.p50  tilt.p50")
    for gname, gd in out.items():
        for b in gd["bands"]:
            f = b["feats"]
            n = f.get("fz_t", {}).get("n", 0)
            if n < 5: continue
            line = (f"{gname:22s}  {b['depth_mm_lo']:>4.0f}-{b['depth_mm_hi']:<4.0f}  {n:>6d}   "
                    f"{f.get('fz_t',{}).get('p50',float('nan')):>+7.2f}    "
                    f"{f.get('F_lat_base',{}).get('p50',float('nan')):>+7.2f}      "
                    f"{f.get('r_cop_m',{}).get('p50',float('nan'))*1000:>+7.2f}      "
                    f"{f.get('dz_dt_mm_s',{}).get('p50',float('nan')):>+7.2f}    "
                    f"{f.get('tilt_deg',{}).get('p50',float('nan')):>+5.2f}")
            print(line)
    print(f"\n(group sizes: {[(k, v['n_episodes']) for k,v in out.items()]})")


if __name__ == "__main__":
    main()
