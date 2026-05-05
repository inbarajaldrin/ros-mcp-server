# Reference: Examine the search-phase motion pattern for FAILURES.
# Question: do failed attempts show spiral motion (algo trying spiral), or a fixed-direction push, or random?
# Compare to GOLD successes.
import json, os, sys, math
import numpy as np
from collections import defaultdict
sys.path.insert(0, os.path.dirname(__file__))
from _paths import DATA_DIR

OUT_DIR = str(DATA_DIR)
SUMMARIES = json.load(open(os.path.join(OUT_DIR, "summaries.json")))


def main():
    targets = [
        ("GOLD u_orange",  lambda r: r["object"]=="u_orange"   and r["outcome"]=="success" and "20260503" in r["csv_path"]),
        ("FAIL u_orange",  lambda r: r["object"]=="u_orange"   and r["outcome"]=="abort"   and "20260504" in r["csv_path"] and r["final_z_drop_mm"] < 5.0),
    ]
    for label, pred in targets:
        print(f"\n=== {label} ===")
        # For each episode, summarize: total xy travel post-contact (path length), bounding box, dominant direction, F_lat avg
        path_lens = []; bbox_diags = []; dom_dirs = []; f_lat_avgs = []; cmd_fzs = []
        finals = []
        durations = []
        for r in SUMMARIES:
            if not pred(r): continue
            ps_path = os.path.join(OUT_DIR,"per_sample",os.path.basename(r["csv_path"]).replace(".csv",".per_sample.json"))
            if not os.path.exists(ps_path): continue
            ps = json.load(open(ps_path))
            ci = r["contact_idx_active"]
            t = np.array(ps["t_s"], dtype=float)
            tcp_x = np.array(ps["tcp_x"], dtype=float)
            tcp_y = np.array(ps["tcp_y"], dtype=float)
            n = len(t)
            # Limit to post-contact window of 10 s (or until end)
            t_end_target = t[ci] + 10.0
            i_end = n
            for i in range(ci, n):
                if t[i] >= t_end_target: i_end = i; break
            xs = tcp_x[ci:i_end]; ys = tcp_y[ci:i_end]
            xs = xs[np.isfinite(xs)]; ys = ys[np.isfinite(ys)]
            if len(xs) < 5 or len(ys) < 5: continue
            # path length
            dxs = np.diff(xs); dys = np.diff(ys)
            path_len_mm = float(np.sum(np.sqrt(dxs*dxs + dys*dys))) * 1000
            # bounding box diagonal
            bbox = math.hypot((xs.max()-xs.min())*1000, (ys.max()-ys.min())*1000)
            # dominant direction (PCA of (xs,ys))
            xs_c = xs - xs.mean(); ys_c = ys - ys.mean()
            cov = np.cov(np.stack([xs_c, ys_c]))
            evals, evecs = np.linalg.eigh(cov)
            v = evecs[:, -1]
            dom_ang = math.degrees(math.atan2(v[1], v[0]))
            # F_lat avg (base) over window
            fxb = np.array(ps["fx_b"][ci:i_end], dtype=float); fyb = np.array(ps["fy_b"][ci:i_end], dtype=float)
            f_mag = np.sqrt(fxb*fxb + fyb*fyb)
            f_mag_avg = float(np.nanmean(f_mag))
            cmd_fz_avg = float(np.nanmean(np.array(ps["commanded_fz"][ci:i_end], dtype=float)))

            path_lens.append(path_len_mm)
            bbox_diags.append(bbox)
            dom_dirs.append(dom_ang)
            f_lat_avgs.append(f_mag_avg)
            cmd_fzs.append(cmd_fz_avg)
            finals.append(r["final_z_drop_mm"])
            durations.append((t[i_end-1] - t[ci]))
        if not path_lens:
            print("  empty"); continue
        print(f"  episodes considered: {len(path_lens)}")
        print(f"  post-contact 10s window summaries (median, IQR):")
        def pq(arr):
            a = np.array(arr)
            return np.percentile(a,50), np.percentile(a,25), np.percentile(a,75)
        for name, arr in [("path_len_mm", path_lens), ("bbox_diag_mm", bbox_diags),
                          ("F_lat_base_N", f_lat_avgs), ("cmd_fz_N", cmd_fzs),
                          ("final_z_drop_mm", finals), ("duration_s", durations)]:
            p50, p25, p75 = pq(arr)
            print(f"    {name:18s}  p25={p25:>+8.2f}  p50={p50:>+8.2f}  p75={p75:>+8.2f}")
        # Path-length / bbox-diag ratio: high = the path winds around (spiral, search), low = direct
        ratios = [pl/bb if bb > 0.1 else 0 for pl, bb in zip(path_lens, bbox_diags)]
        p50, p25, p75 = pq(ratios)
        print(f"    path/bbox_ratio   p25={p25:>+8.2f}  p50={p50:>+8.2f}  p75={p75:>+8.2f}   (>3 = winding/spiral, ~1 = straight push)")


if __name__ == "__main__":
    main()
