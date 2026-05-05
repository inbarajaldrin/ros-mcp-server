# Reference: validate the Fz_t-collapse discriminator and produce a single-figure score per episode.
# Hypothesis: a sustained Fz_t collapse (|Fz_t| < 2 N for >= 0.5 s with dz/dt < -2 mm/s) is the durable transition.
# Score per episode: did we ever achieve durable collapse + how deep did we get afterward.
import json, os, sys, math
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from _paths import DATA_DIR

OUT_DIR = str(DATA_DIR)
SUMMARIES = json.load(open(os.path.join(OUT_DIR, "summaries.json")))


def smooth(arr, w=50):
    n=len(arr); out=np.zeros(n); buf=[]
    for i,v in enumerate(arr):
        if v is not None and v==v:
            buf.append(v)
            if len(buf)>w: buf.pop(0)
            out[i]=sum(buf)/len(buf) if buf else float("nan")
        else:
            out[i]=float("nan")
    return out


def find_durable_collapse(fz_t_smooth, dz_dt_mm_s_smooth, ci, t,
                          fz_th=2.0, dz_th=-2.0, sustain_s=0.5):
    """First post-contact index where |Fz_t| < fz_th AND dz/dt < dz_th sustained for sustain_s."""
    n = len(fz_t_smooth)
    if n == 0: return None
    fs = 1.0 / np.median(np.diff(t[:200])) if len(t) > 1 else 100.0
    sustain_n = int(round(sustain_s * fs))
    run = 0
    for i in range(ci, n):
        v = fz_t_smooth[i]; dz = dz_dt_mm_s_smooth[i]
        if v is None or v != v:
            run = 0; continue
        if abs(v) < fz_th and dz < dz_th:
            run += 1
            if run >= sustain_n:
                return i - run + 1
        else:
            run = 0
    return None


def main():
    rows = []
    for r in SUMMARIES:
        ps_path = os.path.join(OUT_DIR,"per_sample",os.path.basename(r["csv_path"]).replace(".csv",".per_sample.json"))
        if not os.path.exists(ps_path): continue
        ps = json.load(open(ps_path))
        ci = r["contact_idx_active"]
        if ci is None: continue
        t = np.array(ps["t_s"], dtype=float)
        fz_t = np.array(ps["fz_t"], dtype=float)
        dz_dt = np.array(ps["dz_dt_mm_s"], dtype=float)
        fz_t_s = smooth(fz_t.tolist(), 50)
        dz_dt_s = smooth(dz_dt.tolist(), 50)
        ki = find_durable_collapse(fz_t_s, dz_dt_s, ci, t)
        durable = ki is not None
        z_drop_after_collapse_mm = float("nan")
        if durable:
            z_drop_arr = np.array(ps["z_drop_mm"], dtype=float)
            z_drop_after_collapse_mm = float(np.nanmax(z_drop_arr[ki:])) - float(z_drop_arr[ki])
        rows.append({
            "csv": os.path.basename(r["csv_path"]),
            "object": r["object"],
            "outcome": r["outcome"],
            "durable_collapse": durable,
            "final_z_drop_mm": r["final_z_drop_mm"],
            "z_drop_after_collapse_mm": z_drop_after_collapse_mm,
        })

    # Confusion-matrix style: durable_collapse vs (final_z_drop > 20mm = "actually seated")
    from collections import Counter
    cm = Counter()
    for r in rows:
        actually_seated = r["final_z_drop_mm"] > 20.0
        cm[(r["durable_collapse"], actually_seated)] += 1
    print(f"\nConfusion: durable_collapse vs (final_z_drop > 20mm)")
    print(f"  collapse=T, seated=T  -> {cm[(True, True)]}")
    print(f"  collapse=T, seated=F  -> {cm[(True, False)]}")
    print(f"  collapse=F, seated=T  -> {cm[(False, True)]}")
    print(f"  collapse=F, seated=F  -> {cm[(False, False)]}")

    # Group breakdown
    print(f"\nBy group:")
    by_g = {}
    for r in rows:
        # date from csv name
        date = "20260503" if "20260503" in r["csv"] else "20260504"
        g = (r["object"], r["outcome"], date)
        by_g.setdefault(g, []).append(r)
    print(f"{'group':50s}  n   coll_rate   seated_rate   coll∧seated")
    for g, rs in sorted(by_g.items()):
        n = len(rs)
        cr = sum(1 for r in rs if r["durable_collapse"]) / n
        sr = sum(1 for r in rs if r["final_z_drop_mm"]>20) / n
        cs = sum(1 for r in rs if r["durable_collapse"] and r["final_z_drop_mm"]>20) / n
        print(f"  obj={g[0]:20s} out={g[1]:8s} {g[2]}: n={n:>3d}  {cr*100:>5.1f}%      {sr*100:>5.1f}%       {cs*100:>5.1f}%")


if __name__ == "__main__":
    main()
