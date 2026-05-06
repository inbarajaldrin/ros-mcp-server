#!/usr/bin/env python3
# Reference: per-episode export to JSON for the reusable HTML viewer.
# Bundles main CSV + sidecars + meta into a single self-contained JSON the
# viewer (../viewer/compare.html) can load and plot.
#
# Includes proper OBJECT pose computation via the verified cad_lookup chain
# (gripper_center = TCP + R_ee @ (0,0,flange_offset); object = gripper_center - R_object_world @ grasp_offset_in_object).
#
# Usage:
#   python3 22_export_episode_for_viewer.py <basename>
# e.g. python3 22_export_episode_for_viewer.py insert_u_orange_20260505_193645

import csv, json, math, os, sys
import numpy as np
from scipy.spatial.transform import Rotation as R
sys.path.insert(0, os.path.dirname(__file__))
from _paths import LOG_DIR, DATA_DIR

# Try to load the verified grasp offset from cad_lookup (run from repo root for this to work)
_repo_root = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
try:
    from compliant_insertion_studio.wrapper.cad_lookup import (
        load_grasp_offset_in_object,
        GRIPPER_CENTER_TOOL_OFFSET_M,
    )
    HAVE_CAD = True
except Exception as e:
    HAVE_CAD = False
    GRIPPER_CENTER_TOOL_OFFSET_M = 0.2286
    print(f"NOTE: cad_lookup unavailable ({e}); object pose chain will be approximate", file=sys.stderr)


def to_f(s, d=float("nan")):
    try: return float(s)
    except: return d


def col(rows, k):
    return [to_f(r.get(k)) for r in rows]


def load_csv(path):
    if not os.path.exists(path): return []
    with open(path) as fh:
        return list(csv.DictReader(fh))


def quat_to_tilt_deg(qx, qy, qz, qw):
    z_world_z = -(1 - 2 * (qx*qx + qy*qy))
    z_world_z = max(-1.0, min(1.0, z_world_z))
    return math.degrees(math.acos(z_world_z))


def quat_to_yaw_deg(qx, qy, qz, qw):
    return math.degrees(math.atan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz)))


def find_contact_idx(fz_arr, threshold=5.0, sustain=10):
    sm = np.convolve(np.abs(fz_arr), np.ones(5)/5, mode="same")
    above = sm > threshold
    run = 0
    for i, a in enumerate(above):
        run = run + 1 if a else 0
        if run >= sustain: return i - run + 1
    return None


def compute_object_xyz(tcp_xyz, tcp_quat, obj_quat_world, grasp_offset_in_object,
                      flange_offset=GRIPPER_CENTER_TOOL_OFFSET_M):
    """Use the verified cad_lookup math (inverted) to compute object centroid in world."""
    if any(math.isnan(v) for v in tcp_quat) or any(math.isnan(v) for v in obj_quat_world):
        return [float("nan"), float("nan"), float("nan")]
    R_ee = R.from_quat(tcp_quat).as_matrix()
    R_obj_world = R.from_quat(obj_quat_world).as_matrix()
    gripper_center_world = np.array(tcp_xyz) + R_ee @ np.array([0.0, 0.0, flange_offset])
    object_xyz = gripper_center_world - R_obj_world @ np.array(grasp_offset_in_object)
    return object_xyz.tolist()


def main():
    if len(sys.argv) < 2:
        print("usage: 22_export_episode_for_viewer.py <basename>"); sys.exit(2)
    basename = sys.argv[1]
    log_dir = str(LOG_DIR)
    main_csv = os.path.join(log_dir, basename + ".csv")
    meta_path = os.path.join(log_dir, basename + ".meta.json")
    if not os.path.exists(main_csv):
        print(f"main CSV not found: {main_csv}"); sys.exit(2)

    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    rows = load_csv(main_csv)
    rows_active = [r for r in rows if r.get("phase") == "ACTIVE"]
    if not rows_active:
        print("no ACTIVE rows"); sys.exit(2)

    # Pull grasp offset for this object/grasp_id (CAD-verified chain)
    obj_name = meta.get("object", "u_orange")
    grasp_id = int(meta.get("grasp_id", 1))
    grasp_offset = [0.0, 0.0, 0.0]
    if HAVE_CAD:
        try:
            grasp_offset = list(load_grasp_offset_in_object(obj_name, grasp_id))
        except Exception as e:
            print(f"grasp_offset load failed: {e}", file=sys.stderr)

    t = col(rows_active, "t_s")
    px = col(rows_active, "tcp_x"); py = col(rows_active, "tcp_y"); pz = col(rows_active, "tcp_z")
    qx = col(rows_active, "tcp_qx"); qy = col(rows_active, "tcp_qy"); qz = col(rows_active, "tcp_qz"); qw = col(rows_active, "tcp_qw")
    fx = col(rows_active, "fx"); fy = col(rows_active, "fy"); fz = col(rows_active, "fz")
    tx = col(rows_active, "tx"); ty = col(rows_active, "ty"); tz = col(rows_active, "tz")
    obj_qx = col(rows_active, "obj_qx"); obj_qy = col(rows_active, "obj_qy"); obj_qz = col(rows_active, "obj_qz"); obj_qw = col(rows_active, "obj_qw")
    cmd_fz_main = col(rows_active, "commanded_fz")

    # Compute derived per-tick: tilt, yaw, F_lat magnitudes, object xyz
    tilt = [quat_to_tilt_deg(qx[i], qy[i], qz[i], qw[i]) if not math.isnan(qx[i]) else float("nan")
            for i in range(len(rows_active))]
    yaw = [quat_to_yaw_deg(qx[i], qy[i], qz[i], qw[i]) if not math.isnan(qx[i]) else float("nan")
           for i in range(len(rows_active))]
    F_lat_tool = [math.hypot(fx[i] or 0, fy[i] or 0) for i in range(len(rows_active))]
    T_lat_tool = [math.hypot(tx[i] or 0, ty[i] or 0) for i in range(len(rows_active))]

    obj_xyz = []
    for i in range(len(rows_active)):
        if math.isnan(qx[i]) or math.isnan(obj_qx[i]):
            obj_xyz.append([float("nan")] * 3); continue
        try:
            o = compute_object_xyz([px[i], py[i], pz[i]],
                                    [qx[i], qy[i], qz[i], qw[i]],
                                    [obj_qx[i], obj_qy[i], obj_qz[i], obj_qw[i]],
                                    grasp_offset)
            obj_xyz.append(o)
        except Exception:
            obj_xyz.append([float("nan")] * 3)

    # Find contact moment
    fz_arr = np.array([f if f is not None else 0.0 for f in fz])
    ci = find_contact_idx(fz_arr)
    contact_t = t[ci] if ci is not None else None
    contact_xyz = [px[ci], py[ci], pz[ci]] if ci is not None else None
    contact_yaw = yaw[ci] if ci is not None else None

    # Sidecars
    joints = load_csv(os.path.join(log_dir, basename + ".joints_raw.csv"))
    wrench_raw = load_csv(os.path.join(log_dir, basename + ".wrench_raw.csv"))
    cmd_wr = load_csv(os.path.join(log_dir, basename + ".cmd_wrench_raw.csv"))

    # CAD-predicted seat for reference
    pred = meta.get("cad_prediction", {}).get("predicted_tcp_at_seat", {})
    pred_xyz = pred.get("xyz_m")
    pred_quat = pred.get("quat_xyzw")

    out = {
        "basename": basename,
        "object": obj_name,
        "grasp_id": grasp_id,
        "outcome": meta.get("outcome"),
        "outcome_reason": meta.get("outcome_reason"),
        "assist_level": meta.get("assist_level"),
        "schema_version": meta.get("schema_version"),
        "force_mode_params": meta.get("force_mode_params", {}),
        "predicted_tcp_at_seat_m": pred_xyz,
        "predicted_tcp_at_seat_quat": pred_quat,
        "grasp_offset_in_object_m": grasp_offset,
        "flange_offset_m": GRIPPER_CENTER_TOOL_OFFSET_M,
        "contact": {
            "idx": ci,
            "t_s": contact_t,
            "tcp_xyz_m": contact_xyz,
            "tcp_yaw_deg": contact_yaw,
        },
        "active": {
            "n": len(rows_active),
            "t_s": t,
            "tcp_x": px, "tcp_y": py, "tcp_z": pz,
            "tcp_qx": qx, "tcp_qy": qy, "tcp_qz": qz, "tcp_qw": qw,
            "tilt_deg": tilt,
            "yaw_deg": yaw,
            "fx_tool": fx, "fy_tool": fy, "fz_tool": fz,
            "tx_tool": tx, "ty_tool": ty, "tz_tool": tz,
            "F_lat_tool_N": F_lat_tool,
            "T_lat_tool_Nm": T_lat_tool,
            "commanded_fz_main": cmd_fz_main,
            "obj_xyz_m": obj_xyz,
        },
        "joints_raw": {
            "n": len(joints),
            "t_s": [(to_f(r.get("stamp_sec")) + to_f(r.get("stamp_nsec"))/1e9) for r in joints],
            **{f"j{j}_{kind}": [to_f(r.get(f"j{j}_{kind}")) for r in joints]
               for j in range(6) for kind in ("pos", "vel", "eff")},
        } if joints else {"n": 0},
        "wrench_raw": {
            "n": len(wrench_raw),
            "t_s": [(to_f(r.get("stamp_sec")) + to_f(r.get("stamp_nsec"))/1e9) for r in wrench_raw],
            "fx": [to_f(r.get("fx")) for r in wrench_raw],
            "fy": [to_f(r.get("fy")) for r in wrench_raw],
            "fz": [to_f(r.get("fz")) for r in wrench_raw],
            "tx": [to_f(r.get("tx")) for r in wrench_raw],
            "ty": [to_f(r.get("ty")) for r in wrench_raw],
            "tz": [to_f(r.get("tz")) for r in wrench_raw],
        } if wrench_raw else {"n": 0},
        "cmd_wrench_raw": {
            "n": len(cmd_wr),
            "t_s": [to_f(r.get("t_s")) for r in cmd_wr],
            "cmd_fx": [to_f(r.get("cmd_fx")) for r in cmd_wr],
            "cmd_fy": [to_f(r.get("cmd_fy")) for r in cmd_wr],
            "cmd_fz": [to_f(r.get("cmd_fz")) for r in cmd_wr],
            "cmd_tx": [to_f(r.get("cmd_tx")) for r in cmd_wr],
            "cmd_ty": [to_f(r.get("cmd_ty")) for r in cmd_wr],
            "cmd_tz": [to_f(r.get("cmd_tz")) for r in cmd_wr],
            "gain": [to_f(r.get("gain")) for r in cmd_wr],
            "damping": [to_f(r.get("damping")) for r in cmd_wr],
            "source": [r.get("source") for r in cmd_wr],
        } if cmd_wr else {"n": 0},
    }

    out_path = os.path.join(str(DATA_DIR), f"viewer_{basename}.json")
    with open(out_path, "w") as fh:
        json.dump(out, fh, default=lambda x: None if (isinstance(x, float) and math.isnan(x)) else x)
    print(f"written: {out_path}  ({os.path.getsize(out_path)/1024:.0f} KB)")
    print(f"  active samples: {len(rows_active)}")
    print(f"  joints_raw: {out['joints_raw'].get('n', 0)}")
    print(f"  wrench_raw: {out['wrench_raw'].get('n', 0)}")
    print(f"  cmd_wrench_raw: {out['cmd_wrench_raw'].get('n', 0)}")


if __name__ == "__main__":
    main()
