# Reference: Reverse-engineering operator-demo control law for force-compliant peg-in-hole.
# Reads every insert_*.csv + .meta.json, emits per-episode JSON with:
#   - canonical phase boundaries (contact / engagement / seated) derived from physics
#   - per-sample derived features (F_lat, r_cop, z_drop, xy excursion, dz/dt, tilt, ...)
#   - depth-banded aggregates (median + IQR per band, in tool frame AND base frame)
import csv, glob, json, math, os, sys
from collections import defaultdict
import numpy as np
from scipy.spatial.transform import Rotation as R
sys.path.insert(0, os.path.dirname(__file__))
from _paths import LOG_DIR as _LOG_DIR, DATA_DIR as _DATA_DIR, PER_SAMPLE_DIR

LOG_DIR = str(_LOG_DIR)
OUT_DIR = str(_DATA_DIR)

# Depth bands (mm post-contact). 0–25 mm covers full descent.
DEPTH_BANDS_MM = [(0, 1), (1, 2), (2, 5), (5, 10), (10, 15), (15, 20), (20, 30)]

# Smoothing windows
SMOOTH_F_WIN_S = 0.5   # smooth Fz over 0.5 s
SMOOTH_DZ_WIN_S = 2.0  # net z-descent over 2 s window

# Contact threshold
FZ_CONTACT_N = 5.0     # Fz crosses this once peg meets surface


def to_float(s):
    try:
        v = float(s)
        if math.isnan(v):
            return None
        return v
    except Exception:
        return None


def load_csv(path):
    rows = []
    with open(path) as fh:
        rd = csv.DictReader(fh)
        for r in rd:
            rows.append(r)
    return rows


def boxsmooth(x, win):
    if win <= 1: return x.copy()
    n = len(x)
    out = np.empty(n)
    cs = np.concatenate([[0.0], np.cumsum(np.nan_to_num(x, nan=0.0))])
    cnt = np.concatenate([[0.0], np.cumsum(~np.isnan(x))])
    half = win // 2
    for i in range(n):
        a = max(0, i - half); b = min(n, i + half + 1)
        c = cnt[b] - cnt[a]
        out[i] = (cs[b] - cs[a]) / c if c > 0 else float("nan")
    return out


def transform_wrench_tool_to_base(fx, fy, fz, tx, ty, tz, q_xyzw):
    """Sensed wrench in tool0_controller -> base frame.
    Pure rotation: t = R(q) * w_tool. (Measurement, no force/torque coupling.)"""
    rot = R.from_quat(q_xyzw).as_matrix()
    f_base = rot @ np.array([fx, fy, fz])
    t_base = rot @ np.array([tx, ty, tz])
    return f_base, t_base


def analyze_csv(csv_path, meta):
    rows = load_csv(csv_path)
    if len(rows) < 50:
        return None

    # Parse all rows into numpy columns. NaN-filled where values absent.
    n = len(rows)
    fields_f = ["t_s","tcp_x","tcp_y","tcp_z","tcp_qx","tcp_qy","tcp_qz","tcp_qw",
                "target_x","target_y","target_z","fx","fy","fz","tx","ty","tz","commanded_fz",
                "dx","dy","dz","droll","dpitch","dyaw"]
    arr = {f: np.array([to_float(r.get(f, "nan")) for r in rows], dtype=float) for f in fields_f}
    phase = np.array([r.get("phase","") for r in rows])
    hands_off = np.array([to_float(r.get("hands_off","0")) or 0 for r in rows])

    # Identify ACTIVE rows
    active_idx = np.where(phase == "ACTIVE")[0]
    if len(active_idx) < 10:
        return None
    a0, a1 = active_idx[0], active_idx[-1]
    aslice = slice(a0, a1 + 1)
    t = arr["t_s"][aslice].copy()
    t = t - t[0]  # local time within ACTIVE
    n_a = len(t)
    dt = np.diff(t).mean() if n_a > 1 else 0.01
    fs = 1.0 / dt if dt > 0 else 100.0

    # Tool-frame wrench (raw)
    fx_t = arr["fx"][aslice]; fy_t = arr["fy"][aslice]; fz_t = arr["fz"][aslice]
    tx_t = arr["tx"][aslice]; ty_t = arr["ty"][aslice]; tz_t = arr["tz"][aslice]
    cmd_fz = arr["commanded_fz"][aslice]

    # TCP pose
    px = arr["tcp_x"][aslice]; py = arr["tcp_y"][aslice]; pz = arr["tcp_z"][aslice]
    qx = arr["tcp_qx"][aslice]; qy = arr["tcp_qy"][aslice]; qz = arr["tcp_qz"][aslice]; qw = arr["tcp_qw"][aslice]
    tgt_x = arr["target_x"][aslice]; tgt_y = arr["target_y"][aslice]; tgt_z = arr["target_z"][aslice]

    # Transform every wrench sample to base frame
    fx_b = np.zeros(n_a); fy_b = np.zeros(n_a); fz_b = np.zeros(n_a)
    tx_b = np.zeros(n_a); ty_b = np.zeros(n_a); tz_b = np.zeros(n_a)
    for i in range(n_a):
        if not (np.isfinite(qx[i]) and np.isfinite(qy[i]) and np.isfinite(qz[i]) and np.isfinite(qw[i])):
            fx_b[i] = fy_b[i] = fz_b[i] = tx_b[i] = ty_b[i] = tz_b[i] = np.nan
            continue
        try:
            f_b, t_b = transform_wrench_tool_to_base(fx_t[i], fy_t[i], fz_t[i],
                                                    tx_t[i], ty_t[i], tz_t[i],
                                                    [qx[i], qy[i], qz[i], qw[i]])
            fx_b[i], fy_b[i], fz_b[i] = f_b
            tx_b[i], ty_b[i], tz_b[i] = t_b
        except Exception:
            fx_b[i] = fy_b[i] = fz_b[i] = tx_b[i] = ty_b[i] = tz_b[i] = np.nan

    # In tool frame: peg-along-axis force is fz_t. Lateral magnitude = sqrt(fx^2 + fy^2).
    F_lat_tool = np.sqrt(fx_t**2 + fy_t**2)
    F_lat_base = np.sqrt(fx_b**2 + fy_b**2)
    # Lateral angle in base frame (atan2). 0 = +X_world, pi/2 = +Y_world.
    F_lat_dir_base = np.arctan2(fy_b, fx_b)  # NaN where fz_b NaN; here just direction.

    # Center-of-pressure radius (tool frame, per CLAUDE.md): r_cop = ‖(-Ty/Fz, Tx/Fz)‖.
    # Use abs(fz_t) — tool-frame Fz can be near 0 or negative briefly.
    safe_fz = np.where(np.abs(fz_t) > 0.5, fz_t, np.nan)
    cop_x = -ty_t / safe_fz
    cop_y = tx_t / safe_fz
    r_cop = np.sqrt(cop_x**2 + cop_y**2)  # meters (since T/F has units of length)

    # Smoothed signals
    win_f = max(1, int(round(SMOOTH_F_WIN_S * fs)))
    win_dz = max(1, int(round(SMOOTH_DZ_WIN_S * fs)))
    fz_t_smooth = boxsmooth(fz_t, win_f)
    F_lat_base_smooth = boxsmooth(F_lat_base, win_f)
    pz_smooth = boxsmooth(pz, win_f)
    # dz/dt over 2s window
    dz_dt = np.gradient(pz_smooth, t) if n_a > 2 else np.zeros(n_a)

    # Tilt: angle of TCP z-axis from world -z (gripper nominally points down).
    # World gripper-z direction = R(q) @ [0,0,1]. Tilt = angle to [0,0,-1] = pi - acos(z_z).
    tilt_deg = np.full(n_a, np.nan)
    for i in range(n_a):
        if not (np.isfinite(qx[i]) and np.isfinite(qy[i]) and np.isfinite(qz[i]) and np.isfinite(qw[i])):
            continue
        try:
            zw = R.from_quat([qx[i], qy[i], qz[i], qw[i]]).as_matrix()[:, 2]
            zdotdown = -zw[2]
            zdotdown = max(-1.0, min(1.0, zdotdown))
            tilt_deg[i] = math.degrees(math.acos(zdotdown))
        except Exception:
            pass

    # Detect contact: first index where smoothed |fz_t| > FZ_CONTACT_N
    abs_fz_smooth = np.abs(fz_t_smooth)
    contact_mask = abs_fz_smooth > FZ_CONTACT_N
    if not contact_mask.any():
        return None
    ci = int(np.argmax(contact_mask))
    if ci == 0 and not contact_mask[0]:
        return None
    contact_z = pz[ci]
    contact_xy = np.array([px[ci], py[ci]])

    # Post-contact derived
    z_drop = (contact_z - pz)  # positive = peg has descended
    xy_excursion = np.sqrt((px - contact_xy[0])**2 + (py - contact_xy[1])**2)

    # Net z-descent over 2s window
    net_dz_2s = np.zeros(n_a)
    for i in range(n_a):
        a = max(0, i - win_dz)
        net_dz_2s[i] = (pz[a] - pz[i])  # positive = descended in last 2 s

    # Engagement: when z_drop becomes "dominant" (>= 5 mm)
    eng_mask = (np.arange(n_a) >= ci) & (z_drop >= 0.005)
    eng_i = int(np.argmax(eng_mask)) if eng_mask.any() else None
    if eng_i == 0 and not eng_mask[0]:
        eng_i = None

    # Seated: z_drop >= 20 mm AND |dz/dt| < 0.5 mm/s for 1 s sustained AND tilt < 5°
    seated_i = None
    seat_sustain_n = max(1, int(round(1.0 * fs)))
    cond = (z_drop >= 0.020) & (np.abs(dz_dt) < 0.0005) & (np.nan_to_num(tilt_deg, nan=99.0) < 5.0)
    if cond.any():
        # find first index where cond holds for seat_sustain_n consecutive samples
        run = 0
        for i in range(ci, n_a):
            if cond[i]:
                run += 1
                if run >= seat_sustain_n:
                    seated_i = i - run + 1
                    break
            else:
                run = 0

    # Final z_drop achieved
    final_z_drop = float(np.nanmax(z_drop[ci:])) if ci < n_a - 1 else 0.0

    # Per-sample summary table for downstream binning
    per_sample = {
        "t_s": t.tolist(),
        "z_drop_mm": (z_drop * 1000.0).tolist(),
        "xy_excursion_mm": (xy_excursion * 1000.0).tolist(),
        "fx_t": fx_t.tolist(), "fy_t": fy_t.tolist(), "fz_t": fz_t.tolist(),
        "tx_t": tx_t.tolist(), "ty_t": ty_t.tolist(), "tz_t": tz_t.tolist(),
        "fx_b": fx_b.tolist(), "fy_b": fy_b.tolist(), "fz_b": fz_b.tolist(),
        "F_lat_tool": F_lat_tool.tolist(),
        "F_lat_base": F_lat_base.tolist(),
        "F_lat_dir_base_rad": F_lat_dir_base.tolist(),
        "r_cop_m": r_cop.tolist(),
        "dz_dt_mm_s": (dz_dt * 1000.0).tolist(),
        "net_dz_2s_mm": (net_dz_2s * 1000.0).tolist(),
        "tilt_deg": tilt_deg.tolist(),
        "tcp_x": px.tolist(), "tcp_y": py.tolist(), "tcp_z": pz.tolist(),
        "commanded_fz": cmd_fz.tolist(),
    }

    summary = {
        "csv_path": csv_path,
        "object": meta.get("object", "?"),
        "outcome": meta.get("outcome", "?"),
        "outcome_reason": meta.get("outcome_reason", "?"),
        "assist_level": meta.get("assist_level", None),
        "wrapper_version": meta.get("wrapper_version", "?"),
        "n_active": n_a,
        "fs_hz": fs,
        "contact_idx_active": ci,
        "contact_t_s": float(t[ci]),
        "contact_xy_m": [float(contact_xy[0]), float(contact_xy[1])],
        "contact_z_m": float(contact_z),
        "engagement_idx_active": eng_i,
        "engagement_t_s": float(t[eng_i]) if eng_i is not None else None,
        "seated_idx_active": seated_i,
        "seated_t_s": float(t[seated_i]) if seated_i is not None else None,
        "final_z_drop_mm": float(final_z_drop * 1000.0),
        "active_duration_s": float(t[-1]),
    }
    return summary, per_sample


def main():
    csvs = sorted(glob.glob(f"{LOG_DIR}/*.csv"))
    summaries = []
    per_sample_dir = str(PER_SAMPLE_DIR)
    skipped = 0
    for csv_path in csvs:
        meta_path = csv_path.replace(".csv", ".meta.json")
        if not os.path.exists(meta_path):
            skipped += 1; continue
        try:
            meta = json.load(open(meta_path))
        except Exception:
            skipped += 1; continue
        try:
            r = analyze_csv(csv_path, meta)
        except Exception as e:
            print(f"FAIL {csv_path}: {e}", file=sys.stderr); skipped += 1; continue
        if r is None:
            skipped += 1; continue
        summary, per_sample = r
        summaries.append(summary)
        ps_path = os.path.join(per_sample_dir, os.path.basename(csv_path).replace(".csv", ".per_sample.json"))
        with open(ps_path, "w") as fh:
            json.dump(per_sample, fh)
    with open(os.path.join(OUT_DIR, "summaries.json"), "w") as fh:
        json.dump(summaries, fh, indent=2)
    print(f"processed: {len(summaries)}    skipped: {skipped}")


if __name__ == "__main__":
    main()
