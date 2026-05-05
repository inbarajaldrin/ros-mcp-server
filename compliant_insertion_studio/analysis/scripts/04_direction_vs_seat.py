# Reference: For each successful demo, compare the operator's pre-collapse F_lat direction
# against the geometric direction "contact_xy -> seat_xy".
# Hypothesis: the operator pushes the peg toward the slot center.
import json, os, sys, math
import numpy as np
from collections import defaultdict
sys.path.insert(0, os.path.dirname(__file__))
from _paths import DATA_DIR

OUT_DIR = str(DATA_DIR)
SUMMARIES = json.load(open(os.path.join(OUT_DIR, "summaries.json")))

def smooth(arr, w=50):
    n=len(arr); out=[None]*n; buf=[]
    for i,v in enumerate(arr):
        if v is not None and v==v:
            buf.append(v);
            if len(buf)>w: buf.pop(0)
            out[i]=sum(buf)/len(buf)
    return out

def find_collapse(fz_smooth, ci, th=2.0, sustain=20):
    n=len(fz_smooth); run=0
    for i in range(ci,n):
        v=fz_smooth[i]
        if v is None or v!=v: continue
        if abs(v)<th:
            run+=1
            if run>=sustain: return i-run+1
        else: run=0
    return None

GROUPS = {
    "u_orange_GOLD":     lambda r: r["object"]=="u_orange"   and r["outcome"]=="success" and "20260503" in r["csv_path"],
    "u_brown_GOLD":      lambda r: r["object"]=="u_brown"    and r["outcome"]=="success" and "20260503" in r["csv_path"],
    "inv_u_yellow_GOLD": lambda r: r["object"]=="inverted_u_yellow" and r["outcome"]=="success" and "20260503" in r["csv_path"],
    "line_green_GOLD":   lambda r: r["object"]=="line_green" and r["outcome"]=="success" and "20260503" in r["csv_path"],
}

def ang_diff_deg(a, b):
    d = a - b
    while d > 180: d -= 360
    while d < -180: d += 360
    return d

def main():
    for gn, pred in GROUPS.items():
        print(f"\n=== {gn} ===")
        print(f"{'csv':45s}  contact_xy(mm)        seat_xy(mm)         c→s_vec(mm,deg)    F_lat_avg(N,deg)    diff(deg)  pre_xy_dxdy(mm,deg)  excursion_dir(deg)")
        for r in SUMMARIES:
            if not pred(r): continue
            ps_path = os.path.join(OUT_DIR,"per_sample",os.path.basename(r["csv_path"]).replace(".csv",".per_sample.json"))
            if not os.path.exists(ps_path): continue
            ps = json.load(open(ps_path))
            ci = r["contact_idx_active"]
            t = ps["t_s"]
            # Find seat = where z reached its min (end of descent before lift) within ACTIVE
            tcp_x = ps["tcp_x"]; tcp_y = ps["tcp_y"]; tcp_z = ps["tcp_z"]
            n = len(t)
            # Find seated index: deepest point sustained 0.5s
            zarr = np.array(tcp_z, dtype=float)
            min_i = int(np.nanargmin(zarr[ci:])) + ci
            seat_x = tcp_x[min_i]; seat_y = tcp_y[min_i]
            cx = tcp_x[ci]; cy = tcp_y[ci]
            cs_dx = (seat_x - cx)*1000  # mm
            cs_dy = (seat_y - cy)*1000
            cs_mag = math.hypot(cs_dx, cs_dy)
            cs_ang = math.degrees(math.atan2(cs_dy, cs_dx))
            # Find Fz_t collapse (start of descent)
            fz_t_s = smooth(ps["fz_t"], 50)
            ki = find_collapse(fz_t_s, ci)
            if ki is None: continue
            # Pre-collapse 1s window
            tc = t[ki]; t0 = max(t[ci], tc - 1.0)
            i0 = ci
            for i in range(ci,ki):
                if t[i] >= t0: i0 = i; break
            fxb = ps["fx_b"][i0:ki]; fyb = ps["fy_b"][i0:ki]
            fx_avg = np.nanmean(np.array(fxb, dtype=float))
            fy_avg = np.nanmean(np.array(fyb, dtype=float))
            f_mag = math.hypot(fx_avg, fy_avg)
            f_ang = math.degrees(math.atan2(fy_avg, fx_avg))
            # xy displacement during pre-collapse window
            dx_pre = (tcp_x[ki] - tcp_x[i0])*1000
            dy_pre = (tcp_y[ki] - tcp_y[i0])*1000
            pre_mag = math.hypot(dx_pre, dy_pre)
            pre_ang = math.degrees(math.atan2(dy_pre, dx_pre))
            # Excursion contact->collapse
            ex = (tcp_x[ki] - cx)*1000
            ey = (tcp_y[ki] - cy)*1000
            ex_mag = math.hypot(ex, ey)
            ex_ang = math.degrees(math.atan2(ey, ex))
            diff_F_vs_CS = ang_diff_deg(f_ang, cs_ang)
            print(f"{os.path.basename(r['csv_path'])[:43]:45s}  ({cx*1000:+6.1f},{cy*1000:+7.1f})   "
                  f"({seat_x*1000:+6.1f},{seat_y*1000:+7.1f})    "
                  f"({cs_mag:>5.1f},{cs_ang:>+6.0f})   "
                  f"({f_mag:>4.1f},{f_ang:>+6.0f})   "
                  f"{diff_F_vs_CS:>+6.0f}     "
                  f"({pre_mag:>4.1f},{pre_ang:>+6.0f})       "
                  f"({ex_mag:>4.1f},{ex_ang:>+6.0f})")


if __name__ == "__main__":
    main()
