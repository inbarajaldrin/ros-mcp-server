#!/usr/bin/env python3
# Reference: deep ground-truth comparison between GOLD operator + iter-6 close-but-failed.
#
# The current iter-6 reached err=(0.7,-0.4)mm with drop_so_far=0.77mm (vs 1.0mm threshold).
# So PEG IS APPROACHING but not breaking through. What does GOLD do at the same moment
# that my algorithm doesn't?
#
# Comparing across THREE modalities NOT yet examined:
#   1. TCP ORIENTATION (quat) — does operator subtly tilt the EE to engage chamfer?
#   2. F/T feedback (native 500Hz) — what's the chamfer-engagement contact signature?
#   3. Joint states — operator drives motion through which kinematic chain?
#
# Episodes:
#   GOLD  = insert_u_orange_20260505_193645  (operator success, gain=1.0 damp=0.7)
#   FAIL6 = insert_u_orange_20260505_203331  (iter-6, peg got to err=(0.7,-0.4), still aborted)

import csv, math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from _paths import LOG_DIR


def load_csv(path):
    return list(csv.DictReader(open(path)))


def to_f(s, d=float("nan")):
    try: return float(s)
    except: return d


def col(rows, k, dtype=float):
    return np.array([to_f(r.get(k)) for r in rows], dtype=dtype)


def stamp_to_seconds(rows):
    s = np.array([to_f(r.get("stamp_sec")) for r in rows])
    ns = np.array([to_f(r.get("stamp_nsec")) for r in rows])
    return s + ns / 1e9


def find_contact(fz, threshold=5.0, sustain=10):
    sm = np.convolve(np.abs(fz), np.ones(5)/5, mode="same")
    above = sm > threshold
    run = 0
    for i, a in enumerate(above):
        run = run + 1 if a else 0
        if run >= sustain: return i - run + 1
    return None


def tcp_tilt_deg_from_quat(qx, qy, qz, qw):
    """Tilt of EE Z-axis from world -Z (face-down). Returns degrees."""
    # EE Z in world: R(q) @ [0,0,1] = (2(qx*qz + qy*qw), 2(qy*qz - qx*qw), 1 - 2(qx²+qy²))
    z_world = -(1 - 2*(qx*qx + qy*qy))  # negate to get tilt from -Z
    z_world = max(-1.0, min(1.0, z_world))
    return math.degrees(math.acos(z_world))


def quat_yaw_deg(qx, qy, qz, qw):
    """Z-axis rotation about world -Z (yaw of EE)."""
    return math.degrees(math.atan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz)))


def analyze(label, basename, seat_xy=(0.0341, -0.3635)):
    print(f"\n{'='*70}\n=== {label}: {basename}\n{'='*70}")
    main = load_csv(os.path.join(str(LOG_DIR), basename + ".csv"))
    joints = load_csv(os.path.join(str(LOG_DIR), basename + ".joints_raw.csv"))
    wrench_raw = load_csv(os.path.join(str(LOG_DIR), basename + ".wrench_raw.csv"))

    active = [r for r in main if r["phase"] == "ACTIVE"]
    t = col(active, "t_s"); fs = 1.0 / np.median(np.diff(t[:200]))
    px = col(active, "tcp_x"); py = col(active, "tcp_y"); pz = col(active, "tcp_z")
    qx = col(active, "tcp_qx"); qy = col(active, "tcp_qy"); qz = col(active, "tcp_qz"); qw = col(active, "tcp_qw")
    fx_t = col(active, "fx"); fy_t = col(active, "fy"); fz_t = col(active, "fz")
    tx_t = col(active, "tx"); ty_t = col(active, "ty"); tz_t = col(active, "tz")

    ci = find_contact(fz_t)
    if ci is None: print("no contact"); return None
    print(f"contact: idx={ci}/{len(active)} t={t[ci]:.2f}s tcp_z={pz[ci]*1000:.1f}mm")

    # TCP tilt evolution (from quat)
    tilt = np.array([tcp_tilt_deg_from_quat(qx[i], qy[i], qz[i], qw[i]) for i in range(len(active))])
    yaw = np.array([quat_yaw_deg(qx[i], qy[i], qz[i], qw[i]) for i in range(len(active))])
    # tilt at contact + 1, 5, 10 s post-contact
    print(f"\n--- TCP ORIENTATION (deg from canonical face-down) ---")
    for ts_label, ts_dt in [("at contact", 0), ("+1s", 1), ("+5s", 5), ("+10s", 10), ("+15s", 15)]:
        idx = min(ci + int(ts_dt * fs), len(active) - 1)
        print(f"  {ts_label:12s}: tilt={tilt[idx]:5.2f}°  yaw={yaw[idx]:+6.1f}°  "
              f"tcp_xy=({px[idx]*1000:+6.2f},{py[idx]*1000:+7.2f})mm  "
              f"err_to_seat=({(seat_xy[0]-px[idx])*1000:+5.2f},{(seat_xy[1]-py[idx])*1000:+5.2f})mm "
              f"z_drop={(pz[ci]-pz[idx])*1000:+.2f}mm")

    # F/T evolution (tool frame; magnitudes in 100ms windows)
    print(f"\n--- F/T at contact + key moments (tool frame) ---")
    for ts_label, ts_dt in [("at contact", 0), ("+0.1s", 0.1), ("+1s", 1), ("+5s", 5), ("+10s", 10)]:
        idx = min(ci + int(ts_dt * fs), len(active) - 1)
        i0 = max(0, idx - int(0.05 * fs))
        i1 = min(len(active), idx + int(0.05 * fs))
        Fmag = math.hypot(np.median(fx_t[i0:i1]), np.median(fy_t[i0:i1]))
        Fz = np.median(fz_t[i0:i1])
        Tmag = math.hypot(np.median(tx_t[i0:i1]), np.median(ty_t[i0:i1]))
        print(f"  {ts_label:12s}: |F_lat|={Fmag:5.2f}N  fz={Fz:+5.2f}N  |T_lat|={Tmag:.3f}Nm")

    # Joints — focus on j0 (base rotation) + j3-j5 (wrist) which carry operator signature
    if joints:
        # Resample joints onto t_active timebase
        t_jt = stamp_to_seconds(joints)
        t_jt = t_jt - t_jt[0] + t[0]
        jpos = {j: np.interp(t, t_jt, col(joints, f"j{j}_pos")) for j in range(6)}
        jeff = {j: np.interp(t, t_jt, col(joints, f"j{j}_eff")) for j in range(6)}

        print(f"\n--- JOINT POSITIONS (rad) at key moments ---")
        for ts_label, ts_dt in [("at contact", 0), ("+5s", 5), ("+10s", 10), ("+15s", 15)]:
            idx = min(ci + int(ts_dt * fs), len(active) - 1)
            print(f"  {ts_label:12s}: " +
                  "  ".join([f"j{j}={jpos[j][idx]:+.4f}" for j in range(6)]))

        print(f"\n--- JOINT EFFORT (Nm) range over 1s windows post-contact ---")
        for ts_label, ts_dt in [("0-1s", 0), ("4-5s", 4), ("9-10s", 9), ("14-15s", 14)]:
            i0 = min(ci + int(ts_dt * fs), len(active) - 1)
            i1 = min(ci + int((ts_dt + 1) * fs), len(active) - 1)
            print(f"  {ts_label:12s}: " +
                  "  ".join([f"j{j}={np.std(jeff[j][i0:i1]):.3f}" for j in range(6)]) + "  (std)")

    # Distance to seat over time
    print(f"\n--- DISTANCE TO SEAT (mm) over time ---")
    seat_dist = np.sqrt((px - seat_xy[0])**2 + (py - seat_xy[1])**2) * 1000
    for ts_label, ts_dt in [("at contact", 0), ("+1s", 1), ("+3s", 3), ("+5s", 5), ("+7s", 7), ("+10s", 10), ("+12s", 12), ("+15s", 15)]:
        idx = min(ci + int(ts_dt * fs), len(active) - 1)
        print(f"  {ts_label:12s}: dist_to_seat={seat_dist[idx]:5.2f}mm  z_drop={(pz[ci]-pz[idx])*1000:+.2f}mm  tilt={tilt[idx]:.2f}°")

    return {"contact_idx": ci, "tilt": tilt, "seat_dist": seat_dist, "z_drop": (pz[ci] - pz) * 1000,
            "yaw": yaw, "fz_t": fz_t, "fx_t": fx_t, "fy_t": fy_t,
            "tx_t": tx_t, "ty_t": ty_t, "fs": fs}


def diff_close_pass(gold_data, fail_data, gold_name, fail_name):
    """Look at the moment when each came CLOSEST to seat — what was different?"""
    print(f"\n{'='*70}\n=== CLOSE-PASS COMPARISON\n{'='*70}")
    g = gold_data; f = fail_data

    g_ci = g["contact_idx"]; f_ci = f["contact_idx"]
    g_seat = g["seat_dist"]; f_seat = f["seat_dist"]
    g_zd = g["z_drop"]; f_zd = f["z_drop"]
    g_tilt = g["tilt"]; f_tilt = f["tilt"]

    # Find idx of minimum dist to seat post-contact
    g_min_idx = int(g_ci + np.argmin(g_seat[g_ci:g_ci + int(20 * g["fs"])]))
    f_min_idx = int(f_ci + np.argmin(f_seat[f_ci:f_ci + int(20 * f["fs"])]))

    print(f"  GOLD min-dist-to-seat:  {g_seat[g_min_idx]:.2f}mm at t+{(g_min_idx-g_ci)/g['fs']:.1f}s  "
          f"tilt={g_tilt[g_min_idx]:.2f}°  z_drop={g_zd[g_min_idx]:.2f}mm")
    print(f"  FAIL min-dist-to-seat:  {f_seat[f_min_idx]:.2f}mm at t+{(f_min_idx-f_ci)/f['fs']:.1f}s  "
          f"tilt={f_tilt[f_min_idx]:.2f}°  z_drop={f_zd[f_min_idx]:.2f}mm")

    # F/T at min dist moment
    print(f"\n  GOLD F/T at min-dist moment:")
    print(f"    fx={g['fx_t'][g_min_idx]:+5.2f}  fy={g['fy_t'][g_min_idx]:+5.2f}  fz={g['fz_t'][g_min_idx]:+5.2f}N  "
          f"tx={g['tx_t'][g_min_idx]:+.3f}  ty={g['ty_t'][g_min_idx]:+.3f}Nm")
    print(f"  FAIL F/T at min-dist moment:")
    print(f"    fx={f['fx_t'][f_min_idx]:+5.2f}  fy={f['fy_t'][f_min_idx]:+5.2f}  fz={f['fz_t'][f_min_idx]:+5.2f}N  "
          f"tx={f['tx_t'][f_min_idx]:+.3f}  ty={f['ty_t'][f_min_idx]:+.3f}Nm")

    # Tilt and yaw evolution from contact onwards
    print(f"\n  TILT change from contact to t+5s:")
    print(f"    GOLD: {g_tilt[g_ci]:.2f}° → {g_tilt[g_ci + int(5*g['fs'])]:.2f}°")
    print(f"    FAIL: {f_tilt[f_ci]:.2f}° → {f_tilt[f_ci + int(5*f['fs'])]:.2f}°")

    print(f"\n  YAW change from contact to t+5s:")
    print(f"    GOLD: {g['yaw'][g_ci]:+.1f}° → {g['yaw'][g_ci + int(5*g['fs'])]:+.1f}°")
    print(f"    FAIL: {f['yaw'][f_ci]:+.1f}° → {f['yaw'][f_ci + int(5*f['fs'])]:+.1f}°")


def main():
    GOLD = "insert_u_orange_20260505_193645"
    FAIL = "insert_u_orange_20260505_203331"  # iter-6: closest yet

    gd = analyze("GOLD operator", GOLD)
    fd = analyze("FAIL iter-6", FAIL)

    if gd and fd:
        diff_close_pass(gd, fd, GOLD, FAIL)


if __name__ == "__main__":
    main()
