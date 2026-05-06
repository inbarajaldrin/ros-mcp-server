#!/usr/bin/env python3
# Reference: 3-way pose comparison the user has been asking for repeatedly:
#
#   GOLD   = operator-driven success TCP+object pose trajectory (193645)
#   FAIL6  = my closest autonomous fail (iter-6, 203331)
#   PROJ   = CAD-derived predicted_tcp_at_seat chain (where TCP *should* be at seat)
#
# Comparison axes (NOT just z_drop):
#   - TCP xyz at each timepoint vs predicted seat xyz
#   - TCP orientation (tilt + yaw) vs canonical face-down
#   - Object xy via tcp_to_object_transform * grasp_id chain
#   - Cumulative pose error from PROJ over time
#
# Goal: find what the operator's pose trajectory does that the algorithm doesn't,
# in 6-DOF terms, not just z.

import csv, json, math, os, sys
import numpy as np
from scipy.spatial.transform import Rotation as R
sys.path.insert(0, os.path.dirname(__file__))
from _paths import LOG_DIR


def load_csv(path):
    return list(csv.DictReader(open(path)))


def to_f(s, d=float("nan")):
    try: return float(s)
    except: return d


def col(rows, k, dt=float):
    return np.array([to_f(r.get(k)) for r in rows], dtype=dt)


def quat_to_tilt_deg(qx, qy, qz, qw):
    """Tilt of EE Z-axis from world -Z."""
    z_world_z = -(1 - 2 * (qx*qx + qy*qy))  # negate to get tilt from -Z
    z_world_z = max(-1.0, min(1.0, z_world_z))
    return math.degrees(math.acos(z_world_z))


def quat_yaw_deg(qx, qy, qz, qw):
    """Yaw (rotation about EE Z-axis) — for canonical face-down should be ~180°."""
    return math.degrees(math.atan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz)))


def find_contact_idx(fz_arr, threshold=5.0, sustain=10):
    sm = np.convolve(np.abs(fz_arr), np.ones(5)/5, mode="same")
    above = sm > threshold
    run = 0
    for i, a in enumerate(above):
        run = run + 1 if a else 0
        if run >= sustain: return i - run + 1
    return None


def get_predicted_seat(meta_path):
    """Pull the CAD-predicted seat TCP from meta JSON."""
    m = json.load(open(meta_path))
    cad = m.get("cad_prediction", {})
    pred = cad.get("predicted_tcp_at_seat", {})
    xyz = pred.get("xyz_m")
    quat = pred.get("quat_xyzw")
    if xyz and quat:
        return tuple(xyz), tuple(quat)
    return None, None


def get_held_quat_xyzw(meta_path):
    """tcp_to_object rotation from meta."""
    m = json.load(open(meta_path))
    t2o = m.get("tcp_to_object_transform", {})
    return t2o.get("quat_xyzw")


def per_sample_object_pose(rows):
    """Object pose chain: object_quat = R(tcp) * R(tcp_to_object). Object xy = tcp_xy (per schema)."""
    out_x = col(rows, "obj_x")
    out_y = col(rows, "obj_y")
    out_z = col(rows, "obj_z")
    qx = col(rows, "obj_qx"); qy = col(rows, "obj_qy"); qz = col(rows, "obj_qz"); qw = col(rows, "obj_qw")
    return out_x, out_y, out_z, qx, qy, qz, qw


def analyze(label, basename):
    print(f"\n{'='*70}\n=== {label}: {basename}\n{'='*70}")
    main = load_csv(os.path.join(str(LOG_DIR), basename + ".csv"))
    meta_path = os.path.join(str(LOG_DIR), basename + ".meta.json")
    pred_xyz, pred_quat = get_predicted_seat(meta_path)
    held_quat = get_held_quat_xyzw(meta_path)
    if pred_xyz:
        print(f"  PROJECTED seat TCP:  xyz=({pred_xyz[0]*1000:+.2f},{pred_xyz[1]*1000:+.2f},{pred_xyz[2]*1000:+.2f})mm")
        if pred_quat:
            print(f"                       quat={pred_quat}")
    if held_quat:
        print(f"  Held quat (tcp→obj): {held_quat}")

    active = [r for r in main if r.get("phase") == "ACTIVE"]
    if not active:
        print("  no ACTIVE rows"); return None
    t = col(active, "t_s")
    px = col(active, "tcp_x"); py = col(active, "tcp_y"); pz = col(active, "tcp_z")
    qx = col(active, "tcp_qx"); qy = col(active, "tcp_qy"); qz = col(active, "tcp_qz"); qw = col(active, "tcp_qw")
    fz_t = col(active, "fz")
    obj_x, obj_y, obj_z, oqx, oqy, oqz, oqw = per_sample_object_pose(active)

    fs = 1.0 / np.median(np.diff(t[:200]))
    ci = find_contact_idx(fz_t)
    if ci is None:
        print("  no contact"); return None
    print(f"  contact at idx={ci}/{len(active)} t={t[ci]:.2f}s")

    # Per-sample tilt + yaw
    tilt = np.array([quat_to_tilt_deg(qx[i], qy[i], qz[i], qw[i]) for i in range(len(active))])
    yaw = np.array([quat_yaw_deg(qx[i], qy[i], qz[i], qw[i]) for i in range(len(active))])

    obj_tilt = np.array([quat_to_tilt_deg(oqx[i], oqy[i], oqz[i], oqw[i]) if not math.isnan(oqx[i]) else float("nan")
                         for i in range(len(active))])
    obj_yaw = np.array([quat_yaw_deg(oqx[i], oqy[i], oqz[i], oqw[i]) if not math.isnan(oqx[i]) else float("nan")
                        for i in range(len(active))])

    # Distance to projected seat (full xyz)
    if pred_xyz:
        tcp_to_pred_3d = np.sqrt((px-pred_xyz[0])**2 + (py-pred_xyz[1])**2 + (pz-pred_xyz[2])**2) * 1000
        tcp_to_pred_xy = np.sqrt((px-pred_xyz[0])**2 + (py-pred_xyz[1])**2) * 1000
        tcp_to_pred_z  = (pz - pred_xyz[2]) * 1000
    else:
        tcp_to_pred_3d = tcp_to_pred_xy = tcp_to_pred_z = np.full(len(active), float("nan"))

    print(f"\n--- TCP pose vs PROJECTED seat (3D distance) at key moments ---")
    print(f"  {'t':>8s}  {'tcp_xy_to_seat':>15s}  {'tcp_z_to_seat':>14s}  {'tcp_3d_to_seat':>15s}  {'tcp_tilt':>9s}  {'obj_tilt':>9s}  {'tcp_yaw':>9s}")
    for ts_label, ts_dt in [("contact", 0), ("+1s", 1), ("+3s", 3), ("+5s", 5), ("+8s", 8), ("+10s", 10), ("+15s", 15), ("+20s", 20)]:
        idx = min(ci + int(ts_dt * fs), len(active) - 1)
        if idx < 0: continue
        print(f"  {ts_label:>8s}  {tcp_to_pred_xy[idx]:>13.2f}mm  {tcp_to_pred_z[idx]:>+12.2f}mm  {tcp_to_pred_3d[idx]:>13.2f}mm  "
              f"{tilt[idx]:>7.2f}°  {obj_tilt[idx]:>7.2f}°  {yaw[idx]:>+7.1f}°")

    # Find moment of MIN 3D distance to projected seat
    seg = slice(ci, min(ci + int(25 * fs), len(active)))
    if seg.stop - seg.start > 10 and not np.all(np.isnan(tcp_to_pred_3d[seg])):
        min_idx = ci + int(np.nanargmin(tcp_to_pred_3d[seg]))
        print(f"\n  MIN 3D dist to PROJECTED seat: {tcp_to_pred_3d[min_idx]:.2f}mm at t+{(min_idx-ci)/fs:.1f}s")
        print(f"    tcp_pose=({px[min_idx]*1000:+.2f},{py[min_idx]*1000:+.2f},{pz[min_idx]*1000:+.2f})mm  "
              f"tilt={tilt[min_idx]:.2f}°  yaw={yaw[min_idx]:+.1f}°")
        if pred_xyz:
            print(f"    pred_seat=({pred_xyz[0]*1000:+.2f},{pred_xyz[1]*1000:+.2f},{pred_xyz[2]*1000:+.2f})mm")
        print(f"    obj_quat=(qx,qy,qz,qw)=({oqx[min_idx]:+.4f},{oqy[min_idx]:+.4f},{oqz[min_idx]:+.4f},{oqw[min_idx]:+.4f})")

    return {
        "label": label, "ci": ci, "fs": fs,
        "px": px, "py": py, "pz": pz, "tilt": tilt, "yaw": yaw,
        "pred_xyz": pred_xyz,
        "tcp_to_pred_3d": tcp_to_pred_3d, "tcp_to_pred_xy": tcp_to_pred_xy, "tcp_to_pred_z": tcp_to_pred_z,
        "obj_tilt": obj_tilt, "obj_yaw": obj_yaw,
    }


def diff_summary(g, f):
    if not (g and f): return
    print(f"\n{'='*70}\n=== 3-WAY DIFF SUMMARY (GOLD vs FAIL vs PROJECTED)\n{'='*70}")
    fs_g = g["fs"]; fs_f = f["fs"]; ci_g = g["ci"]; ci_f = f["ci"]

    # PROJECTED TCP at seat (CAD prior) vs each run's actual at-seat moment
    if g.get("pred_xyz"):
        pg = g["pred_xyz"]
        print(f"\n  PROJECTED TCP at seat:  xyz=({pg[0]*1000:+.2f},{pg[1]*1000:+.2f},{pg[2]*1000:+.2f})mm")

    # min-dist moment for each
    seg_g = slice(ci_g, min(ci_g + int(25*fs_g), len(g["pz"])))
    seg_f = slice(ci_f, min(ci_f + int(25*fs_f), len(f["pz"])))
    g_min = ci_g + int(np.nanargmin(g["tcp_to_pred_3d"][seg_g]))
    f_min = ci_f + int(np.nanargmin(f["tcp_to_pred_3d"][seg_f]))

    print(f"\n  GOLD  min-3D-to-PROJ:  {g['tcp_to_pred_3d'][g_min]:.2f}mm at t+{(g_min-ci_g)/fs_g:.1f}s   "
          f"tcp_xyz=({g['px'][g_min]*1000:+.2f},{g['py'][g_min]*1000:+.2f},{g['pz'][g_min]*1000:+.2f})mm  tilt={g['tilt'][g_min]:.2f}°")
    print(f"  FAIL  min-3D-to-PROJ:  {f['tcp_to_pred_3d'][f_min]:.2f}mm at t+{(f_min-ci_f)/fs_f:.1f}s   "
          f"tcp_xyz=({f['px'][f_min]*1000:+.2f},{f['py'][f_min]*1000:+.2f},{f['pz'][f_min]*1000:+.2f})mm  tilt={f['tilt'][f_min]:.2f}°")

    # tilt evolution from contact to seat
    print(f"\n  TILT evolution (contact → +5s → +10s → +15s):")
    print(f"    GOLD: {g['tilt'][ci_g]:.2f}° → {g['tilt'][ci_g+int(5*fs_g)]:.2f}° → "
          f"{g['tilt'][ci_g+int(10*fs_g)]:.2f}° → {g['tilt'][ci_g+int(15*fs_g)]:.2f}°")
    print(f"    FAIL: {f['tilt'][ci_f]:.2f}° → {f['tilt'][ci_f+int(5*fs_f)]:.2f}° → "
          f"{f['tilt'][ci_f+int(10*fs_f)]:.2f}° → {f['tilt'][ci_f+int(15*fs_f)]:.2f}°")

    # OBJECT tilt (different from TCP tilt — peg may tilt differently than gripper)
    print(f"\n  OBJECT tilt evolution (peg orientation vs canonical face-down):")
    print(f"    GOLD: {g['obj_tilt'][ci_g]:.2f}° → {g['obj_tilt'][ci_g+int(5*fs_g)]:.2f}° → "
          f"{g['obj_tilt'][ci_g+int(10*fs_g)]:.2f}° → {g['obj_tilt'][ci_g+int(15*fs_g)]:.2f}°")
    print(f"    FAIL: {f['obj_tilt'][ci_f]:.2f}° → {f['obj_tilt'][ci_f+int(5*fs_f)]:.2f}° → "
          f"{f['obj_tilt'][ci_f+int(10*fs_f)]:.2f}° → {f['obj_tilt'][ci_f+int(15*fs_f)]:.2f}°")

    # YAW evolution — does operator twist the peg about Z?
    print(f"\n  TCP YAW evolution (angle about EE Z-axis vs canonical):")
    print(f"    GOLD: {g['yaw'][ci_g]:+.2f}° → {g['yaw'][ci_g+int(5*fs_g)]:+.2f}° → "
          f"{g['yaw'][ci_g+int(10*fs_g)]:+.2f}° → {g['yaw'][ci_g+int(15*fs_g)]:+.2f}°")
    print(f"    FAIL: {f['yaw'][ci_f]:+.2f}° → {f['yaw'][ci_f+int(5*fs_f)]:+.2f}° → "
          f"{f['yaw'][ci_f+int(10*fs_f)]:+.2f}° → {f['yaw'][ci_f+int(15*fs_f)]:+.2f}°")


def main():
    GOLD = "insert_u_orange_20260505_193645"
    FAIL = "insert_u_orange_20260505_203331"  # iter-6 closest

    gd = analyze("GOLD operator", GOLD)
    fd = analyze("FAIL iter-6", FAIL)
    diff_summary(gd, fd)


if __name__ == "__main__":
    main()
