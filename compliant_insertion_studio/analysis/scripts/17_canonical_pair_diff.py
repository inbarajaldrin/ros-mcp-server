#!/usr/bin/env python3
# Reference: discovery iteration 008 — canonical A/B pair multi-modality diff
# Purpose: with matched controller/compliance/algo-cmd, what does the operator do that the
#   algorithm doesn't? Uses schema v1.2 sidecars (joints_raw, wrench_raw, cmd_wrench_raw, fm_events)
#   that prior episodes lacked.
#
# Pair:
#   GOLD  insert_u_orange_20260505_193645  outcome=success    (operator-driven)
#   FAIL  insert_u_orange_20260505_193941  outcome=abort      (autonomous, FIND_HOLE STUCK)
# Both at gain_scaling=1.0, damping_factor=0.7.

import csv, json, math, os, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from _paths import LOG_DIR, ANALYSIS_DIR, DATA_DIR

GOLD_BASE = "insert_u_orange_20260505_193645"
FAIL_BASE = "insert_u_orange_20260505_193941"

CONTACT_FZ_THRESH_N = 5.0
CONTACT_SUSTAIN_S = 0.10
WINDOW_PRE_S = 1.0
WINDOW_POST_S = 60.0

# Empirical admittance from FINDINGS (operator demos): 1 N -> 0.5 mm/s steady-state lateral
ADMITTANCE_LIN_M_PER_NS = 0.0005


# -------- I/O helpers --------

def _read_csv_dict(path):
    with open(path) as f:
        rdr = csv.DictReader(f)
        cols = {k: [] for k in rdr.fieldnames}
        for row in rdr:
            for k, v in row.items():
                cols[k].append(v)
    out = {}
    for k, vals in cols.items():
        # try float; fall back to string
        try:
            out[k] = np.array([float(v) if v not in ("", "nan") else np.nan for v in vals])
        except ValueError:
            out[k] = np.array(vals, dtype=object)
    return out


def _stamp_to_t(stamp_sec, stamp_nsec):
    return stamp_sec.astype(np.float64) + stamp_nsec.astype(np.float64) * 1e-9


def load_episode(base):
    main = _read_csv_dict(LOG_DIR / f"{base}.csv")
    joints = _read_csv_dict(LOG_DIR / f"{base}.joints_raw.csv")
    wrench = _read_csv_dict(LOG_DIR / f"{base}.wrench_raw.csv")
    cmd = _read_csv_dict(LOG_DIR / f"{base}.cmd_wrench_raw.csv")
    fm = _read_csv_dict(LOG_DIR / f"{base}.fm_events.csv")
    meta = json.load(open(LOG_DIR / f"{base}.meta.json"))

    # Convert sidecar stamps to wall time
    joints["t_wall"] = _stamp_to_t(joints["stamp_sec"], joints["stamp_nsec"])
    wrench["t_wall"] = _stamp_to_t(wrench["stamp_sec"], wrench["stamp_nsec"])

    # Anchor: main CSV t_s starts at episode-relative 0; sidecars are wall time.
    # Use the wrench callback as the anchor: the first wrench_raw row corresponds to roughly
    # the moment force_mode started (after PRE/HOVER/ZERO complete). We instead align via
    # cmd_wrench_raw: the first 'start_force_mode' event's t_s is on the main-CSV clock,
    # and the corresponding wall time is the wrench_raw row at-or-just-after that.
    # For our analysis we mostly diff in the post-contact window, anchor on contact within
    # each modality independently.
    return {"main": main, "joints": joints, "wrench": wrench, "cmd": cmd, "fm": fm, "meta": meta}


# -------- Feature computation --------

def _smooth(x, win):
    """Centered moving average. win in samples."""
    if win < 2: return x
    k = np.ones(int(win)) / float(int(win))
    pad = int(win) // 2
    xp = np.concatenate([np.full(pad, x[0] if x.size else 0.0), x, np.full(pad, x[-1] if x.size else 0.0)])
    return np.convolve(xp, k, mode="valid")[: len(x)]


def find_contact_idx_main(main):
    """First index where ACTIVE phase + smoothed |fz| > 5N for >= 100ms."""
    t = main["t_s"].astype(float)
    fz = main["fz"].astype(float)
    phase = main["phase"]
    # |fz_t| -- tool frame; the schema notes a vertical peg makes contact Fz POSITIVE in tool frame.
    fz_abs = np.abs(fz)
    # smoothing: 100Hz -> 100ms = 10 samples
    fz_s = _smooth(fz_abs, 10)
    over = fz_s > CONTACT_FZ_THRESH_N
    # Restrict to ACTIVE
    in_active = phase == "ACTIVE"
    sustain_n = max(1, int(round(CONTACT_SUSTAIN_S * 100)))
    for i in range(len(t) - sustain_n):
        if not in_active[i]: continue
        if np.all(over[i:i + sustain_n]):
            return i
    return None


def find_contact_idx_wrench_raw(wrench):
    """In native-rate wrench, find first |fz|>5N sustained 100ms = ~50 samples at 500Hz."""
    fz = wrench["fz"].astype(float)
    t = wrench["t_wall"]
    fz_s = _smooth(np.abs(fz), 50)
    over = fz_s > CONTACT_FZ_THRESH_N
    sustain_n = max(1, int(round(CONTACT_SUSTAIN_S * 500)))
    for i in range(len(t) - sustain_n):
        if np.all(over[i:i + sustain_n]):
            return i
    return None


def quat_to_tilt_deg(qx, qy, qz, qw):
    """Tilt = angle between TCP -Z and world -Z. With nominally face-down (q≈(0,1,0,0)),
    tilt deviation is angle between R*[0,0,1] and [0,0,-1] (or between R*[0,0,-1] and [0,0,-1]).
    We compute angle of TCP z-axis from world z-axis."""
    qx = np.asarray(qx); qy = np.asarray(qy); qz = np.asarray(qz); qw = np.asarray(qw)
    # TCP z-axis in world: R * [0,0,1]
    zx = 2 * (qx * qz + qy * qw)
    zy = 2 * (qy * qz - qx * qw)
    zz = 1 - 2 * (qx * qx + qy * qy)
    # Angle from world -Z axis (since gripper points down) -> angle from [0,0,-1]
    cos_a = -zz  # (zx*0 + zy*0 + zz*(-1))
    cos_a = np.clip(cos_a, -1.0, 1.0)
    return np.degrees(np.arccos(cos_a))


def transform_wrench_to_base(fx, fy, fz, qx, qy, qz, qw):
    """Apply rotation R(q) to vector [fx,fy,fz]."""
    # Quaternion (xyzw) rotation of vector v: v' = q*v*q_conj
    fx = np.asarray(fx); fy = np.asarray(fy); fz = np.asarray(fz)
    qx = np.asarray(qx); qy = np.asarray(qy); qz = np.asarray(qz); qw = np.asarray(qw)
    # Use direct rotation matrix
    xx = qx * qx; yy = qy * qy; zz = qz * qz
    xy = qx * qy; xz = qx * qz; yz = qy * qz
    wx = qw * qx; wy = qw * qy; wz = qw * qz
    R00 = 1 - 2 * (yy + zz); R01 = 2 * (xy - wz);     R02 = 2 * (xz + wy)
    R10 = 2 * (xy + wz);     R11 = 1 - 2 * (xx + zz); R12 = 2 * (yz - wx)
    R20 = 2 * (xz - wy);     R21 = 2 * (yz + wx);     R22 = 1 - 2 * (xx + yy)
    bx = R00 * fx + R01 * fy + R02 * fz
    by = R10 * fx + R11 * fy + R12 * fz
    bz = R20 * fx + R21 * fy + R22 * fz
    return bx, by, bz


def compute_main_features(ep):
    m = ep["main"]
    t = m["t_s"].astype(float)
    contact_i = find_contact_idx_main(m)

    # Tilt
    tilt = quat_to_tilt_deg(m["tcp_qx"], m["tcp_qy"], m["tcp_qz"], m["tcp_qw"])

    # Wrench in base frame
    fx_b, fy_b, fz_b = transform_wrench_to_base(
        m["fx"], m["fy"], m["fz"], m["tcp_qx"], m["tcp_qy"], m["tcp_qz"], m["tcp_qw"]
    )
    F_lat_base = np.hypot(fx_b, fy_b)
    F_lat_tool = np.hypot(m["fx"].astype(float), m["fy"].astype(float))
    F_lat_dir_base = np.arctan2(fy_b, fx_b)  # radians

    # r_cop = ‖(-Ty/Fz, Tx/Fz)‖ in tool frame
    fz_t = m["fz"].astype(float)
    tx = m["tx"].astype(float); ty = m["ty"].astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        r_cop = np.where(np.abs(fz_t) > 1.0, np.hypot(-ty / fz_t, tx / fz_t), np.nan)

    # dz/dt smoothed 0.5s (50 samples)
    z = m["tcp_z"].astype(float)
    dt = np.diff(t, prepend=t[0])
    dt[dt <= 0] = np.median(dt[dt > 0]) if np.any(dt > 0) else 0.01
    dz_dt = np.gradient(z, t)
    dz_dt_smooth = _smooth(dz_dt, 50) * 1000.0  # mm/s

    # TCP xy velocity (operator nudge surrogate when cmd_lateral=0)
    vx_tcp = np.gradient(m["tcp_x"].astype(float), t)
    vy_tcp = np.gradient(m["tcp_y"].astype(float), t)
    v_xy = np.hypot(vx_tcp, vy_tcp)

    return {
        "t": t,
        "contact_i": contact_i,
        "tcp_x": m["tcp_x"].astype(float),
        "tcp_y": m["tcp_y"].astype(float),
        "tcp_z": z,
        "tilt_deg": tilt,
        "fx_t": m["fx"].astype(float), "fy_t": m["fy"].astype(float), "fz_t": fz_t,
        "tx_t": tx, "ty_t": ty, "tz_t": m["tz"].astype(float),
        "fx_b": fx_b, "fy_b": fy_b, "fz_b": fz_b,
        "F_lat_base": F_lat_base, "F_lat_tool": F_lat_tool,
        "F_lat_dir_base_rad": F_lat_dir_base,
        "r_cop_m": r_cop,
        "dz_dt_mm_s": dz_dt_smooth,
        "v_xy_m_s": v_xy,
        "vx_tcp": vx_tcp, "vy_tcp": vy_tcp,
        "phase": m["phase"],
        "commanded_fz": m["commanded_fz"].astype(float),
    }


# -------- Cross-modal features --------

def joint_signals(ep):
    j = ep["joints"]
    t = j["t_wall"]
    pos = np.column_stack([j[f"j{i}_pos"].astype(float) for i in range(6)])
    vel = np.column_stack([j[f"j{i}_vel"].astype(float) for i in range(6)])
    eff = np.column_stack([j[f"j{i}_eff"].astype(float) for i in range(6)])
    return {"t": t, "pos": pos, "vel": vel, "eff": eff}


def native_wrench(ep):
    w = ep["wrench"]
    t = w["t_wall"]
    return {"t": t,
            "fx": w["fx"].astype(float), "fy": w["fy"].astype(float), "fz": w["fz"].astype(float),
            "tx": w["tx"].astype(float), "ty": w["ty"].astype(float), "tz": w["tz"].astype(float)}


# -------- Diff computation --------

def depth_band_medians(feat_main, contact_i, surface_z, bands_mm):
    """For each post-contact depth band, return median of every feature."""
    if contact_i is None: return None
    # surface_z = tcp_z at contact (treat the contact-instant z as surface)
    sz = feat_main["tcp_z"][contact_i]
    z = feat_main["tcp_z"][contact_i:]
    z_drop_mm = (sz - z) * 1000.0
    out = {}
    for lo, hi in bands_mm:
        mask = (z_drop_mm >= lo) & (z_drop_mm < hi)
        if not np.any(mask):
            out[f"{lo}-{hi}"] = None
            continue
        idx = np.where(mask)[0] + contact_i
        rec = {}
        for k in ("tilt_deg", "fz_t", "F_lat_base", "F_lat_tool", "r_cop_m",
                 "dz_dt_mm_s", "v_xy_m_s"):
            v = feat_main[k][idx]
            v = v[np.isfinite(v)]
            rec[k] = float(np.median(v)) if v.size else None
        rec["n_samples"] = int(mask.sum())
        rec["depth_med_mm"] = float(np.median(z_drop_mm[mask]))
        out[f"{lo}-{hi}"] = rec
    return out


def cross_correlation(a, b, max_lag):
    """Normalized x-corr; returns (best_lag, best_corr)."""
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a = a - np.nanmean(a); b = b - np.nanmean(b)
    a = np.where(np.isnan(a), 0.0, a); b = np.where(np.isnan(b), 0.0, b)
    sa = np.std(a); sb = np.std(b)
    if sa < 1e-9 or sb < 1e-9: return (0, 0.0)
    a /= sa; b /= sb
    n = min(len(a), len(b))
    a = a[:n]; b = b[:n]
    best = (0, -2.0)
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            c = np.mean(a[-lag:n] * b[:n + lag])
        elif lag > 0:
            c = np.mean(a[:n - lag] * b[lag:n])
        else:
            c = np.mean(a * b)
        if c > best[1]: best = (lag, c)
    return best


def time_of_divergence(t_g, x_g, t_f, x_f, band_quantile=(0.25, 0.75)):
    """When does FAIL leave GOLD's [p25,p75] band, with respect to t=0 contact-aligned?
    We compute on the post-contact window. Here both are the same single trace, so we
    return the first time |x_g - x_f| > 1.5 * IQR(x_g[pre]) over post-contact."""
    # Use simple approach: align both onto a common grid t=[0, 5s] post-contact.
    if t_g is None or t_f is None: return None
    grid = np.arange(0.0, 5.0, 0.01)
    # both inputs are post-contact arrays starting at t=0
    g = np.interp(grid, t_g, x_g, left=np.nan, right=np.nan)
    f = np.interp(grid, t_f, x_f, left=np.nan, right=np.nan)
    diff = np.abs(g - f)
    # Threshold: 1.5x IQR of the GOLD trace itself (proxy for "outside band")
    g_fin = g[np.isfinite(g)]
    if g_fin.size < 5: return None
    p25, p75 = np.percentile(g_fin, [25, 75])
    thr = max(1e-6, 1.5 * (p75 - p25))
    over = np.where((np.isfinite(diff)) & (diff > thr))[0]
    if over.size == 0: return None
    return float(grid[over[0]])


def operator_nudge_signature(feat, contact_i, max_post_s=5.0):
    """Estimate operator's contribution to TCP velocity after subtracting what admittance
    would have produced from the sensed wrench.

    v_admittance_pred = ADMITTANCE * F_lat_base   (toward the side opposite the sensed force)
                                                  i.e. the controller pushes BACK against contact.
    operator_residual = v_observed - (-ADMITTANCE * F_lat_base) = v_observed + ADMITTANCE*F_lat
                        (sign convention: sensor reads opposite of motion direction; if F_y>0
                         due to contact pushing on +Y, controller commands a pushback in -Y;
                         hence under admittance v_y ≈ -ADMITTANCE*F_y. Operator residual is the
                         difference from this prediction.)
    """
    if contact_i is None: return None
    end_i = min(len(feat["t"]) - 1, contact_i + int(max_post_s / 0.01))
    t = feat["t"][contact_i:end_i] - feat["t"][contact_i]
    vx = feat["vx_tcp"][contact_i:end_i]
    vy = feat["vy_tcp"][contact_i:end_i]
    fx_b = feat["fx_b"][contact_i:end_i]
    fy_b = feat["fy_b"][contact_i:end_i]
    # Predicted velocity from sensed wrench (under admittance): operator's hand
    # is the residual not explained by F_sensed.
    # Note: For a hands-free autonomous case, sensed F_lat is all reaction and v should ≈
    # admittance * cmd - admittance * F_react. With cmd_lateral≈0, v_pred ≈ -admittance * F_react.
    vx_pred = -ADMITTANCE_LIN_M_PER_NS * fx_b
    vy_pred = -ADMITTANCE_LIN_M_PER_NS * fy_b
    res_x = vx - vx_pred
    res_y = vy - vy_pred
    res_mag = np.hypot(res_x, res_y) * 1000.0  # mm/s
    return {
        "t_post": t.tolist(),
        "residual_mag_mm_s": res_mag.tolist(),
        "median_residual_mm_s": float(np.nanmedian(res_mag)),
        "p75_residual_mm_s": float(np.nanpercentile(res_mag, 75)),
        "max_residual_mm_s": float(np.nanmax(res_mag)),
        "median_v_xy_observed_mm_s": float(np.nanmedian(np.hypot(vx, vy)) * 1000.0),
        "median_F_lat_base_N": float(np.nanmedian(np.hypot(fx_b, fy_b))),
        "median_residual_dir_rad": float(np.nanmedian(np.arctan2(res_y, res_x))),
    }


# -------- Main --------

def main():
    print("Loading episodes...")
    gold = load_episode(GOLD_BASE)
    fail = load_episode(FAIL_BASE)

    print("Computing main-CSV features...")
    g_feat = compute_main_features(gold)
    f_feat = compute_main_features(fail)

    print(f"GOLD contact_i = {g_feat['contact_i']} (t={g_feat['t'][g_feat['contact_i']]:.2f}s)")
    print(f"FAIL contact_i = {f_feat['contact_i']} (t={f_feat['t'][f_feat['contact_i']]:.2f}s)")

    # Find native-rate contact (for high-freq transient comparison)
    g_w_contact = find_contact_idx_wrench_raw(gold["wrench"])
    f_w_contact = find_contact_idx_wrench_raw(fail["wrench"])

    # Depth bands — main feature (per the prompt)
    bands = [(0, 1), (1, 2), (2, 5), (5, 15), (15, 30), (30, 60), (60, 120)]
    print("Depth-banded medians...")
    g_depth = depth_band_medians(g_feat, g_feat["contact_i"], None, bands)
    f_depth = depth_band_medians(f_feat, f_feat["contact_i"], None, bands)

    # Operator-nudge signature
    print("Operator-nudge signature...")
    g_nudge = operator_nudge_signature(g_feat, g_feat["contact_i"], max_post_s=5.0)
    f_nudge = operator_nudge_signature(f_feat, f_feat["contact_i"], max_post_s=5.0)

    # Cross-correlations: joint_effort_j_i vs tcp_velocity_axis. Need to interpolate tcp velocities
    # onto the joint clock. Joint clock is wall time; main clock is episode-relative.
    print("Joint vs TCP cross-correlations...")
    g_joints = joint_signals(gold); f_joints = joint_signals(fail)

    def joint_vs_tcp_xcor(feat, joints, contact_i):
        if contact_i is None or joints["t"].size < 100: return {}
        # Map main clock to wall via wrench raw alignment: take the first
        # wrench_raw stamp as the wall-time anchor for main t_s=joint_anchor
        # Simple approach: use joints' own time, sliding correlation of effort vs joint velocity
        # And later compute correlation of effort with d/dt(forward kinematics z), but we only need
        # the heuristic: which joint's effort tracks lateral xy motion.
        # Compute: for the post-contact 5s window in main, get vx_tcp, vy_tcp; for each joint
        # find effort over the same wall window; resample, correlate.
        t_main_post = feat["t"][contact_i:contact_i + 500]  # 5s at 100Hz
        if t_main_post.size < 100: return {}
        # Use joints time axis; assume main t=0 corresponds to joints t=joints["t"][0]
        # (both start with the wrapper). Apply offset = joints[0]
        joints_t0 = joints["t"][0]
        t_main_wall = t_main_post + joints_t0
        # Find joint indices in this wall window
        i0 = np.searchsorted(joints["t"], t_main_wall[0])
        i1 = np.searchsorted(joints["t"], t_main_wall[-1])
        if i1 - i0 < 50: return {}
        t_j_win = joints["t"][i0:i1]
        # Resample tcp velocities onto joint clock
        vx_re = np.interp(t_j_win, t_main_wall, feat["vx_tcp"][contact_i:contact_i + len(t_main_post)])
        vy_re = np.interp(t_j_win, t_main_wall, feat["vy_tcp"][contact_i:contact_i + len(t_main_post)])
        v_xy_re = np.hypot(vx_re, vy_re)
        out = {}
        for j in range(6):
            eff = joints["eff"][i0:i1, j]
            jvel = joints["vel"][i0:i1, j]
            lag, corr = cross_correlation(eff, v_xy_re, max_lag=25)
            lag_jv, corr_jv = cross_correlation(jvel, v_xy_re, max_lag=25)
            out[f"j{j}"] = {
                "xcor_eff_vs_v_xy": {"lag": lag, "corr": float(corr)},
                "xcor_vel_vs_v_xy": {"lag": lag_jv, "corr": float(corr_jv)},
                "median_eff_Nm": float(np.median(eff)),
                "abs_max_eff_Nm": float(np.max(np.abs(eff))),
            }
        return out

    g_jx = joint_vs_tcp_xcor(g_feat, g_joints, g_feat["contact_i"])
    f_jx = joint_vs_tcp_xcor(f_feat, f_joints, f_feat["contact_i"])

    # Admittance check: F_lat_base vs tcp_xy_velocity should be opposite-signed in pure admittance.
    # If GOLD shows residual same-sign or larger-than-predicted velocity = operator override.
    print("Admittance residual check...")

    def admittance_check(feat, contact_i, max_post_s=5.0):
        if contact_i is None: return None
        end_i = min(len(feat["t"]) - 1, contact_i + int(max_post_s / 0.01))
        fx_b = feat["fx_b"][contact_i:end_i]; fy_b = feat["fy_b"][contact_i:end_i]
        vx = feat["vx_tcp"][contact_i:end_i]; vy = feat["vy_tcp"][contact_i:end_i]
        # Pearson correlation, with negative correlation = admittance-consistent.
        def pearson(a, b):
            mask = np.isfinite(a) & np.isfinite(b)
            a = a[mask]; b = b[mask]
            if a.size < 10: return None
            return float(np.corrcoef(a, b)[0, 1])
        return {
            "corr_Fx_vx": pearson(fx_b, vx),
            "corr_Fy_vy": pearson(fy_b, vy),
            # Operator override metric: |v_observed| vs |v_admittance_pred|
            "ratio_v_obs_to_v_pred": float(
                np.nanmedian(np.hypot(vx, vy)) /
                max(1e-9, np.nanmedian(np.hypot(ADMITTANCE_LIN_M_PER_NS * fx_b,
                                                ADMITTANCE_LIN_M_PER_NS * fy_b)))
            ),
        }

    g_adm = admittance_check(g_feat, g_feat["contact_i"], 5.0)
    f_adm = admittance_check(f_feat, f_feat["contact_i"], 5.0)

    # cmd identity check
    g_cmd = gold["cmd"]; f_cmd = fail["cmd"]
    cmd_check = {
        "GOLD_n_cmd_events": int(len(g_cmd["t_s"])),
        "FAIL_n_cmd_events": int(len(f_cmd["t_s"])),
        "GOLD_unique_cmds": [
            {"cmd_fx": float(g_cmd["cmd_fx"][i]), "cmd_fy": float(g_cmd["cmd_fy"][i]),
             "cmd_fz": float(g_cmd["cmd_fz"][i]), "gain": float(g_cmd["gain"][i]),
             "damping": float(g_cmd["damping"][i]), "src": str(g_cmd["source"][i])}
            for i in range(min(8, len(g_cmd["t_s"])))
        ],
        "FAIL_unique_cmds": [
            {"cmd_fx": float(f_cmd["cmd_fx"][i]), "cmd_fy": float(f_cmd["cmd_fy"][i]),
             "cmd_fz": float(f_cmd["cmd_fz"][i]), "gain": float(f_cmd["gain"][i]),
             "damping": float(f_cmd["damping"][i]), "src": str(f_cmd["source"][i])}
            for i in range(min(12, len(f_cmd["t_s"])))
        ],
    }

    # Time-of-divergence per feature
    print("Time-of-divergence...")

    def post_contact_arr(feat, key, contact_i, max_post_s=5.0):
        if contact_i is None: return None, None
        end_i = min(len(feat["t"]) - 1, contact_i + int(max_post_s / 0.01))
        t = feat["t"][contact_i:end_i] - feat["t"][contact_i]
        x = feat[key][contact_i:end_i]
        return t, x

    div = {}
    for key in ("tilt_deg", "F_lat_base", "F_lat_tool", "r_cop_m", "dz_dt_mm_s", "v_xy_m_s",
                "fz_t", "tcp_x", "tcp_y", "tcp_z"):
        tg, xg = post_contact_arr(g_feat, key, g_feat["contact_i"])
        tf, xf = post_contact_arr(f_feat, key, f_feat["contact_i"])
        div[key] = time_of_divergence(tg, xg, tf, xf)

    # Headline numbers post-contact (1s window)
    def window_stats(feat, contact_i, t_lo, t_hi):
        if contact_i is None: return None
        i0 = contact_i + int(t_lo / 0.01)
        i1 = contact_i + int(t_hi / 0.01)
        i1 = min(i1, len(feat["t"]) - 1)
        out = {}
        for k in ("tilt_deg", "F_lat_base", "F_lat_tool", "r_cop_m",
                  "dz_dt_mm_s", "v_xy_m_s", "fz_t"):
            v = feat[k][i0:i1]
            v = v[np.isfinite(v)]
            out[k] = {
                "median": float(np.median(v)) if v.size else None,
                "max": float(np.max(np.abs(v))) if v.size else None,
            }
        # XY excursion in window
        i_c = contact_i
        dx = feat["tcp_x"][i0:i1] - feat["tcp_x"][i_c]
        dy = feat["tcp_y"][i0:i1] - feat["tcp_y"][i_c]
        out["xy_excursion_mm"] = {
            "max": float(np.max(np.hypot(dx, dy)) * 1000) if dx.size else None,
            "end": float(np.hypot(dx[-1], dy[-1]) * 1000) if dx.size else None,
        }
        # z drop
        dz = feat["tcp_z"][i_c] - feat["tcp_z"][i0:i1]
        out["z_drop_mm"] = {
            "max": float(np.max(dz) * 1000) if dz.size else None,
            "end": float(dz[-1] * 1000) if dz.size else None,
        }
        return out

    headline = {
        "GOLD_first_1s": window_stats(g_feat, g_feat["contact_i"], 0.0, 1.0),
        "FAIL_first_1s": window_stats(f_feat, f_feat["contact_i"], 0.0, 1.0),
        "GOLD_first_2s": window_stats(g_feat, g_feat["contact_i"], 0.0, 2.0),
        "FAIL_first_2s": window_stats(f_feat, f_feat["contact_i"], 0.0, 2.0),
        "GOLD_first_5s": window_stats(g_feat, g_feat["contact_i"], 0.0, 5.0),
        "FAIL_first_5s": window_stats(f_feat, f_feat["contact_i"], 0.0, 5.0),
    }

    # Final outputs
    out = {
        "pair": {"GOLD": GOLD_BASE, "FAIL": FAIL_BASE},
        "meta": {
            "GOLD_outcome": gold["meta"]["outcome"], "GOLD_reason": gold["meta"]["outcome_reason"],
            "FAIL_outcome": fail["meta"]["outcome"], "FAIL_reason": fail["meta"]["outcome_reason"],
            "matched_force_mode_params": gold["meta"]["force_mode_params"] == fail["meta"]["force_mode_params"],
            "GOLD_force_mode": gold["meta"]["force_mode_params"],
            "FAIL_force_mode": fail["meta"]["force_mode_params"],
        },
        "contact": {
            "GOLD_main_contact_t_s": float(g_feat["t"][g_feat["contact_i"]]) if g_feat["contact_i"] is not None else None,
            "FAIL_main_contact_t_s": float(f_feat["t"][f_feat["contact_i"]]) if f_feat["contact_i"] is not None else None,
            "GOLD_native_wrench_contact_idx": g_w_contact,
            "FAIL_native_wrench_contact_idx": f_w_contact,
        },
        "cmd_identity": cmd_check,
        "depth_band_medians_GOLD": g_depth,
        "depth_band_medians_FAIL": f_depth,
        "headline_windows": headline,
        "operator_nudge_signature_GOLD": {k: v for k, v in g_nudge.items() if k not in ("t_post", "residual_mag_mm_s")} if g_nudge else None,
        "operator_nudge_signature_FAIL": {k: v for k, v in f_nudge.items() if k not in ("t_post", "residual_mag_mm_s")} if f_nudge else None,
        "admittance_check_GOLD": g_adm,
        "admittance_check_FAIL": f_adm,
        "joint_vs_tcp_xcor_GOLD": g_jx,
        "joint_vs_tcp_xcor_FAIL": f_jx,
        "time_of_divergence_per_feature_s": div,
    }

    out_path = DATA_DIR / "canonical_pair_diff.json"
    json.dump(out, open(out_path, "w"), indent=2, default=str)
    print(f"Wrote {out_path}")

    # Headline print
    print()
    print("===== HEADLINE FINDINGS =====")
    print(f"GOLD outcome: {gold['meta']['outcome']} ({gold['meta']['outcome_reason']})")
    print(f"FAIL outcome: {fail['meta']['outcome']} ({fail['meta']['outcome_reason']})")
    print(f"force_mode_params matched: {gold['meta']['force_mode_params'] == fail['meta']['force_mode_params']}")
    print()
    print("Post-contact 1s window medians:")
    for label, key, unit in [
        ("F_lat_base (median)", "F_lat_base", "N"),
        ("F_lat_tool (median)", "F_lat_tool", "N"),
        ("v_xy (median)", "v_xy_m_s", "m/s"),
        ("|fz_t| (median)", "fz_t", "N"),
        ("tilt_deg (median)", "tilt_deg", "deg"),
        ("r_cop (median)", "r_cop_m", "m"),
    ]:
        gv = headline["GOLD_first_1s"][key]["median"] if headline["GOLD_first_1s"] else None
        fv = headline["FAIL_first_1s"][key]["median"] if headline["FAIL_first_1s"] else None
        print(f"  {label:30s} GOLD={gv}  FAIL={fv}  ({unit})")

    print()
    print("Post-contact 1s xy excursion (mm):")
    print(f"  GOLD max={headline['GOLD_first_1s']['xy_excursion_mm']['max']:.3f}  end={headline['GOLD_first_1s']['xy_excursion_mm']['end']:.3f}")
    print(f"  FAIL max={headline['FAIL_first_1s']['xy_excursion_mm']['max']:.3f}  end={headline['FAIL_first_1s']['xy_excursion_mm']['end']:.3f}")

    print()
    print("Post-contact 2s z drop (mm):")
    print(f"  GOLD max={headline['GOLD_first_2s']['z_drop_mm']['max']:.3f}  end={headline['GOLD_first_2s']['z_drop_mm']['end']:.3f}")
    print(f"  FAIL max={headline['FAIL_first_2s']['z_drop_mm']['max']:.3f}  end={headline['FAIL_first_2s']['z_drop_mm']['end']:.3f}")

    print()
    print("Operator-nudge signature (TCP velocity not explained by sensed-wrench admittance):")
    if g_nudge:
        print(f"  GOLD median residual={g_nudge['median_residual_mm_s']:.3f} mm/s   p75={g_nudge['p75_residual_mm_s']:.3f}   max={g_nudge['max_residual_mm_s']:.3f}")
    if f_nudge:
        print(f"  FAIL median residual={f_nudge['median_residual_mm_s']:.3f} mm/s   p75={f_nudge['p75_residual_mm_s']:.3f}   max={f_nudge['max_residual_mm_s']:.3f}")

    print()
    print("Admittance check (corr(F,v) — negative is admittance-consistent):")
    if g_adm: print(f"  GOLD: corr(Fx,vx)={g_adm['corr_Fx_vx']}, corr(Fy,vy)={g_adm['corr_Fy_vy']}, ratio_v_obs/v_pred={g_adm['ratio_v_obs_to_v_pred']:.2f}")
    if f_adm: print(f"  FAIL: corr(Fx,vx)={f_adm['corr_Fx_vx']}, corr(Fy,vy)={f_adm['corr_Fy_vy']}, ratio_v_obs/v_pred={f_adm['ratio_v_obs_to_v_pred']:.2f}")

    print()
    print("Time-of-divergence (s post-contact, when FAIL leaves GOLD's IQR-band):")
    for k, v in div.items():
        print(f"  {k:18s} {v}")

    print()
    print("Joint xcor (effort vs |v_xy|, post-contact 5s):")
    if g_jx and f_jx:
        for j in range(6):
            ge = g_jx.get(f"j{j}", {}).get("xcor_eff_vs_v_xy", {})
            fe = f_jx.get(f"j{j}", {}).get("xcor_eff_vs_v_xy", {})
            g_abs = g_jx.get(f"j{j}", {}).get("abs_max_eff_Nm")
            f_abs = f_jx.get(f"j{j}", {}).get("abs_max_eff_Nm")
            print(f"  j{j}: GOLD corr={ge.get('corr')} lag={ge.get('lag')}  |eff|max={g_abs}   "
                  f"FAIL corr={fe.get('corr')} lag={fe.get('lag')}  |eff|max={f_abs}")

    print()
    print("FAIL cmd_wrench events (should show FSM lateral-search activity):")
    for c in cmd_check["FAIL_unique_cmds"]:
        print(f"  {c}")

    return out


if __name__ == "__main__":
    main()
