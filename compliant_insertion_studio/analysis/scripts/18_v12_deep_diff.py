#!/usr/bin/env python3
# Reference: deep multi-modality raw-data diff between GOLD operator demo and FAIL autonomous
# at matched compliance (gain=1.0 damp=0.7). Uses the v1.2 sidecars that were never analyzed before.
#
# Three questions, four sub-analyses:
#
#   Q1 — DIRECTION OF IMPACT
#     For every contact event (sample where Fz_t crosses ±5N), compute:
#       - direction of TCP velocity at that moment (in base frame)
#       - direction of sensed F_lat in base frame
#       - direction from contact-xy toward seat-xy (geometric prior)
#     Compare GOLD vs FAIL: do operators consistently push toward seat? Does algo?
#
#   Q2 — FEEDBACK RESPONSE
#     For each TCP-velocity sample, compute the lag-correlated relationship to:
#       - sensed F_lat (admittance: F_in → v_out, lag τ_admittance)
#       - cmd F_lat (algo intent → motion, lag τ_command)
#       - operator-residual = v_TCP - admittance(F_sensed) (the part neither cmd nor admittance explains)
#     This shows where the operator's hand contributes.
#
#   Q3 — JOINT vs TCP CAUSAL CHAIN
#     Cross-correlate joint torques j0..j5 with TCP velocity components.
#     Which joints carry which TCP motions? Operator's signature should differ from algo's.
#     Specifically: is GOLD's TCP xy motion driven by j0 (base) or by j3-j5 (wrist)?
#
# All output: data/v12_deep_diff.json — machine-readable.
# Also prints headline numbers to stdout.

import csv, json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from _paths import LOG_DIR, DATA_DIR


def load_main_csv(basename: str):
    """Returns list of dicts from the main 100Hz CSV."""
    path = os.path.join(str(LOG_DIR), basename + ".csv")
    return list(csv.DictReader(open(path)))


def load_sidecar_csv(basename: str, suffix: str):
    """Returns list of dicts from a sidecar CSV (joints_raw / wrench_raw / cmd_wrench_raw)."""
    path = os.path.join(str(LOG_DIR), basename + "." + suffix + ".csv")
    return list(csv.DictReader(open(path)))


def to_float(s, default=float("nan")):
    try:
        v = float(s)
        return v if not math.isnan(v) else default
    except (TypeError, ValueError):
        return default


def column(rows, key, dtype=float):
    return np.array([to_float(r.get(key)) for r in rows], dtype=dtype)


def stamp_to_seconds(rows):
    """Convert stamp_sec / stamp_nsec columns to wall seconds (relative)."""
    s = np.array([to_float(r.get("stamp_sec")) for r in rows])
    ns = np.array([to_float(r.get("stamp_nsec")) for r in rows])
    return s + ns / 1e9


def find_contact_idx(fz_arr, threshold=5.0, sustain_n=10):
    """First sample where smoothed |fz| > threshold for sustain_n consecutive samples."""
    abs_fz = np.abs(fz_arr)
    # smooth by 50ms window (assume ~100Hz so 5 samples)
    win = 5
    if len(abs_fz) < win:
        return None
    sm = np.convolve(abs_fz, np.ones(win) / win, mode="same")
    above = sm > threshold
    # find first run of `sustain_n` consecutive True
    run = 0
    for i, a in enumerate(above):
        run = run + 1 if a else 0
        if run >= sustain_n:
            return i - run + 1
    return None


def smooth(arr, w):
    if w <= 1:
        return arr.copy()
    n = len(arr)
    out = np.empty(n)
    cs = np.concatenate([[0.0], np.cumsum(np.nan_to_num(arr, nan=0.0))])
    cnt = np.concatenate([[0.0], np.cumsum(~np.isnan(arr))])
    half = w // 2
    for i in range(n):
        a = max(0, i - half)
        b = min(n, i + half + 1)
        c = cnt[b] - cnt[a]
        out[i] = (cs[b] - cs[a]) / c if c > 0 else float("nan")
    return out


def direction_deg(x, y):
    """atan2 → degrees in [-180, 180]."""
    return math.degrees(math.atan2(y, x))


def lag_correlation(x, y, max_lag=50):
    """Returns (best_lag_samples, best_corr). Positive lag = y leads x."""
    n = min(len(x), len(y))
    x = x[:n] - np.nanmean(x[:n])
    y = y[:n] - np.nanmean(y[:n])
    x = np.nan_to_num(x); y = np.nan_to_num(y)
    sx = np.std(x); sy = np.std(y)
    if sx < 1e-9 or sy < 1e-9:
        return 0, 0.0
    best_lag = 0
    best_corr = 0.0
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            xr, yr = x[lag:], y[: n - lag]
        else:
            xr, yr = x[: n + lag], y[-lag:]
        if len(xr) < 10:
            continue
        c = np.corrcoef(xr, yr)[0, 1]
        if not math.isnan(c) and abs(c) > abs(best_corr):
            best_corr = c
            best_lag = lag
    return best_lag, best_corr


def analyze_episode(basename: str, label: str, seat_xy_m=(0.0341, -0.3635)):
    print(f"\n=== {label}: {basename} ===")
    main = load_main_csv(basename)
    joints = load_sidecar_csv(basename, "joints_raw")
    wrench = load_sidecar_csv(basename, "wrench_raw")
    cmd_wr = load_sidecar_csv(basename, "cmd_wrench_raw")

    # Restrict main CSV to ACTIVE phase
    active = [r for r in main if r.get("phase") == "ACTIVE"]
    t_active = np.array([to_float(r["t_s"]) for r in active])
    fx_t = column(active, "fx"); fy_t = column(active, "fy"); fz_t = column(active, "fz")
    tx_t = column(active, "tx"); ty_t = column(active, "ty"); tz_t = column(active, "tz")
    px = column(active, "tcp_x"); py = column(active, "tcp_y"); pz = column(active, "tcp_z")
    cmd_fz_main = column(active, "commanded_fz")

    # Find contact (in main CSV at 100Hz)
    ci = find_contact_idx(fz_t)
    if ci is None:
        print(f"  no contact detected (active rows={len(active)})")
        return None
    t_contact = t_active[ci]
    print(f"  contact_idx={ci}/{len(active)} at t={t_contact:.2f}s, contact_z={pz[ci]*1000:.1f}mm")

    # ----- Q1: DIRECTION OF IMPACT -----
    contact_xy = (px[ci], py[ci])
    # Geometric direction contact -> seat
    dx_geom = seat_xy_m[0] - contact_xy[0]
    dy_geom = seat_xy_m[1] - contact_xy[1]
    geom_dist_mm = math.hypot(dx_geom, dy_geom) * 1000
    geom_dir = direction_deg(dx_geom, dy_geom)
    print(f"  Q1 contact_xy=({contact_xy[0]*1000:+.1f},{contact_xy[1]*1000:+.1f})mm,  "
          f"contact→seat = {geom_dist_mm:.1f}mm @ {geom_dir:+.0f}°")

    # In 1s post-contact, what direction does TCP move?
    fs = 1.0 / np.median(np.diff(t_active[:200])) if len(t_active) > 1 else 100.0
    n_1s = int(fs * 1.0)
    end1s = min(ci + n_1s, len(active) - 1)
    if end1s > ci:
        d_tcp_x = (px[end1s] - px[ci]) * 1000
        d_tcp_y = (py[end1s] - py[ci]) * 1000
        tcp_dir = direction_deg(d_tcp_x, d_tcp_y)
        tcp_dist = math.hypot(d_tcp_x, d_tcp_y)
        # Mean F_lat direction in tool frame (in 1s post-contact)
        fxs = fx_t[ci:end1s]; fys = fy_t[ci:end1s]
        fx_mean = np.nanmean(fxs); fy_mean = np.nanmean(fys)
        f_lat_dir_tool = direction_deg(fx_mean, fy_mean)
        f_lat_mag = math.hypot(fx_mean, fy_mean)
        print(f"  Q1 1s-post-contact:  tcp_disp={tcp_dist:.2f}mm @ {tcp_dir:+.0f}°  "
              f"F_lat_tool_avg={f_lat_mag:.2f}N @ {f_lat_dir_tool:+.0f}°")

    # ----- Q2: FEEDBACK RESPONSE — TCP velocity correlations -----
    # Compute velocities (numerical)
    vx = np.gradient(px, t_active) * 1000  # mm/s
    vy = np.gradient(py, t_active) * 1000
    f_lat_mag_t = np.sqrt(fx_t**2 + fy_t**2)
    v_xy_mag = np.sqrt(vx**2 + vy**2)

    # In 5s post-contact, how do |F_lat| and |v_xy| correlate (admittance check)?
    n_5s = int(fs * 5.0)
    end5s = min(ci + n_5s, len(active))
    seg = slice(ci, end5s)
    if end5s - ci > 50:
        # sm both
        f_lat_s = smooth(f_lat_mag_t[seg], 10)
        v_xy_s = smooth(v_xy_mag[seg], 10)
        lag, corr = lag_correlation(v_xy_s, f_lat_s, max_lag=int(fs * 0.5))
        # mean values
        med_f = np.nanmedian(f_lat_s)
        med_v = np.nanmedian(v_xy_s)
        print(f"  Q2 5s-post-contact: |F_lat|.med={med_f:.2f}N  |v_xy|.med={med_v:.2f}mm/s  "
              f"corr(|v_xy|↔|F_lat|)={corr:+.2f} @ lag={lag/fs*1000:+.0f}ms")

    # ----- Q3: JOINT-TORQUE → TCP-VELOCITY -----
    # Joints sidecar at native rate (~250Hz); resample to main CSV's 100Hz time axis
    if len(joints) > 50:
        t_jt = stamp_to_seconds(joints)
        # Align: use first stamp as t=0, then offset to align with main CSV's t_s
        t_jt = t_jt - t_jt[0] + t_active[0]
        # For each joint, get effort
        joint_corr = {}
        for j in range(6):
            eff = np.array([to_float(r.get(f"j{j}_eff")) for r in joints], dtype=float)
            # Resample to main CSV's t_active
            eff_resampled = np.interp(t_active, t_jt, eff, left=np.nan, right=np.nan)
            # In 5s post-contact, correlate eff with |v_xy|
            seg_eff = eff_resampled[seg]
            seg_v = v_xy_mag[seg]
            valid = np.isfinite(seg_eff) & np.isfinite(seg_v)
            if valid.sum() < 50:
                continue
            seg_eff = smooth(seg_eff[valid], 10)
            seg_v = smooth(seg_v[valid], 10)
            lag, corr = lag_correlation(seg_v, seg_eff, max_lag=int(fs * 0.3))
            peak_eff = float(np.nanmax(np.abs(seg_eff)))
            joint_corr[f"j{j}"] = {"corr_v_xy": corr, "lag_ms": lag / fs * 1000, "peak_eff_Nm": peak_eff}
        # Print ranked by |corr|
        ranked = sorted(joint_corr.items(), key=lambda kv: -abs(kv[1]["corr_v_xy"]))
        print(f"  Q3 joint→TCP_xy correlation (ranked):")
        for jk, jv in ranked[:6]:
            print(f"    {jk}: corr={jv['corr_v_xy']:+.2f}  lag={jv['lag_ms']:+.0f}ms  peak_eff={jv['peak_eff_Nm']:.2f}Nm")

    # ----- Q4: COMMANDED vs SENSED — what did the controller actually do? -----
    if len(cmd_wr) > 5:
        cmd_t = column(cmd_wr, "t_s")
        cmd_fx = column(cmd_wr, "cmd_fx"); cmd_fy = column(cmd_wr, "cmd_fy")
        cmd_fz = column(cmd_wr, "cmd_fz")
        # Restrict to start_force_mode events with non-zero selection
        active_cmd_mask = np.array([r.get("source") == "start_force_mode" for r in cmd_wr])
        # During FIND_HOLE (post-contact, 1-15s)
        post_mask = (cmd_t >= t_active[ci] + 0.5) & (cmd_t <= t_active[ci] + 15.0) & active_cmd_mask
        if post_mask.sum() > 5:
            sub_fx = cmd_fx[post_mask]; sub_fy = cmd_fy[post_mask]
            cmd_lat_mag = np.sqrt(sub_fx**2 + sub_fy**2)
            print(f"  Q4 cmd_F_lat in [+0.5s, +15s] post-contact: n={post_mask.sum()}, "
                  f"med={np.median(cmd_lat_mag):.2f}N  p25={np.percentile(cmd_lat_mag,25):.2f}N  p75={np.percentile(cmd_lat_mag,75):.2f}N")
            # Direction stats
            cmd_dirs = np.array([direction_deg(fx, fy) for fx, fy in zip(sub_fx, sub_fy)])
            # circular stats
            sx = np.sum(np.cos(np.radians(cmd_dirs))); sy = np.sum(np.sin(np.radians(cmd_dirs)))
            mean_dir = math.degrees(math.atan2(sy, sx))
            R = math.sqrt(sx**2 + sy**2) / len(cmd_dirs)
            print(f"  Q4 cmd_F_lat direction: mean={mean_dir:+.0f}°, concentration R={R:.2f}  (R→1=fixed, R→0=spinning)")

    # Return summary for json export
    return {
        "label": label,
        "basename": basename,
        "n_active": len(active),
        "contact_idx": int(ci),
        "contact_t_s": float(t_contact),
        "contact_xy_m": [float(contact_xy[0]), float(contact_xy[1])],
        "contact_z_m": float(pz[ci]),
        "geom_to_seat_mm": float(geom_dist_mm),
        "geom_to_seat_dir_deg": float(geom_dir),
        "tcp_disp_1s_mm": float(tcp_dist) if "tcp_dist" in dir() else None,
        "tcp_dir_1s_deg": float(tcp_dir) if "tcp_dir" in dir() else None,
        "f_lat_tool_1s_med_N": float(f_lat_mag) if "f_lat_mag" in dir() else None,
        "f_lat_tool_1s_dir_deg": float(f_lat_dir_tool) if "f_lat_dir_tool" in dir() else None,
        "feedback_5s": {
            "f_lat_mag_med": float(med_f) if "med_f" in dir() else None,
            "v_xy_mag_med": float(med_v) if "med_v" in dir() else None,
            "corr_v_F": float(corr) if "corr" in dir() else None,
            "lag_ms": float(lag / fs * 1000) if "lag" in dir() and "fs" in dir() else None,
        },
        "joint_corr_v_xy": joint_corr if "joint_corr" in dir() else {},
    }


def main():
    GOLD = "insert_u_orange_20260505_193645"
    FAIL = "insert_u_orange_20260505_193941"
    seat_xy = (0.0341, -0.3635)

    res = {}
    for basename, label in [(GOLD, "GOLD operator"), (FAIL, "FAIL autonomous")]:
        try:
            r = analyze_episode(basename, label, seat_xy_m=seat_xy)
            res[label] = r
        except Exception as e:
            print(f"FAIL on {basename}: {e}")
            import traceback; traceback.print_exc()

    out = os.path.join(str(DATA_DIR), "v12_deep_diff.json")
    with open(out, "w") as fh:
        json.dump(res, fh, indent=2, default=str)
    print(f"\nwritten: {out}")

    # Headline diff
    print(f"\n{'=' * 60}\nHEADLINE diff (GOLD vs FAIL):\n{'=' * 60}")
    g = res.get("GOLD operator"); f = res.get("FAIL autonomous")
    if g and f:
        if g.get("tcp_disp_1s_mm") and f.get("tcp_disp_1s_mm"):
            print(f"  TCP disp 1s post-contact:  GOLD={g['tcp_disp_1s_mm']:.2f}mm  FAIL={f['tcp_disp_1s_mm']:.2f}mm")
            print(f"  TCP direction post-contact: GOLD={g['tcp_dir_1s_deg']:+.0f}°  FAIL={f['tcp_dir_1s_deg']:+.0f}°  geom→seat={g['geom_to_seat_dir_deg']:+.0f}°")
        if g.get("feedback_5s") and f.get("feedback_5s"):
            ff = f["feedback_5s"]; gg = g["feedback_5s"]
            print(f"  |F_lat|.med 5s:  GOLD={gg.get('f_lat_mag_med'):.2f}N  FAIL={ff.get('f_lat_mag_med'):.2f}N")
            print(f"  |v_xy|.med 5s:  GOLD={gg.get('v_xy_mag_med'):.2f}mm/s  FAIL={ff.get('v_xy_mag_med'):.2f}mm/s")
            print(f"  corr(v↔F):  GOLD={gg.get('corr_v_F'):+.2f}  FAIL={ff.get('corr_v_F'):+.2f}")
        # Joint diff
        jg = g.get("joint_corr_v_xy", {}); jf = f.get("joint_corr_v_xy", {})
        if jg and jf:
            print(f"  joint→v_xy strongest:")
            for j in ["j0", "j3", "j4", "j5"]:
                if j in jg and j in jf:
                    print(f"    {j}: GOLD corr={jg[j]['corr_v_xy']:+.2f} eff={jg[j]['peak_eff_Nm']:.2f}Nm  vs  "
                          f"FAIL corr={jf[j]['corr_v_xy']:+.2f} eff={jf[j]['peak_eff_Nm']:.2f}Nm")


if __name__ == "__main__":
    main()
