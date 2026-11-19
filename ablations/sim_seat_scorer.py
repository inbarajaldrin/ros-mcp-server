#!/usr/bin/env python3
"""Ground-truth seat scorer and part observer for the DroidStack MuJoCo sim replay.

    python3 -m ablations.sim_seat_scorer --score u_brown,u_orange --base base1
    python3 -m ablations.sim_seat_scorer --observe u_brown,line_green --base base1

SIM ONLY: it reads the simulator's ground-truth stream /objects_poses_sim and
refuses to run unless ROS_MCP_MODE=sim.

Ruling A (lead, 2026-11-19): a part's seat is judged on its FINAL pose, after
the gripper has opened and the arm has retracted, once the part has settled.
The bar is the R22 one: position within 1 mm and orientation within 1 deg of
the seated CAD pose (assembly JSON component pose relative to the base's ground
truth, orientation taken over the part's fold-symmetric equivalents, as
translate_object aims).  The pose while the jaws still held the part and the
contact-phase pad shift are reported by translate_object and are not gated.
The replay scores each part before the next one starts and re-scores every
seated part at the end of the run, so a later insert that disturbs an earlier
part is caught.

--drop <part> (place-down, lead approval 2026-11-19): the held part's lowest CAD
point above the table from ground truth, at the end of a place-down descent.  A
place-down ends at move_down's floor exactly as on the real arm; for a sideways
stance that floor puts the gripper centre (the grasp point)
TABLE_COLLISION_MARGIN_SIDEWAYS above the table (replay_real_assembly.
_move_down_floor_z), so no part can legitimately be held higher than that at the
floor - a larger drop means the descent did not end where the real one would, and
is refused.

"Settled" means: at least --settle-min-s of simulator time has passed since the
call started, and over the trailing --steady-s window every scored part moved
less than 0.05 mm and 0.05 deg.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TOPIC = "/objects_poses_sim"
SEAT_BAR_MM = 1.0
SEAT_BAR_DEG = 1.0
STEADY_MM = 0.05
STEADY_DEG = 0.05
RULING = "A (lead, 2026-11-19): final pose after release + retract, settled; 1 mm / 1 deg"


def _matrix(translation, rotation_xyzw):
    T = np.eye(4)
    T[:3, :3] = R.from_quat(rotation_xyzw).as_matrix()
    T[:3, 3] = translation
    return T


def seat_error(T_object, T_base, component, fold_data):
    """(mm, deg) of a part's pose from its seated pose on the base."""
    from primitives.shared.fold_symmetry import equivalent_orientations
    p = component["position"]
    q = component["rotation"]["quaternion"]
    target_position = T_base[:3, 3] + T_base[:3, :3] @ np.array([p["x"], p["y"], p["z"]])
    position_mm = float(np.linalg.norm(T_object[:3, 3] - target_position) * 1000.0)
    R_relative = T_base[:3, :3].T @ T_object[:3, :3]
    R_target = R.from_quat([q["x"], q["y"], q["z"], q["w"]]).as_matrix()
    angle = min(float(R.from_matrix(R_relative.T @ R_eq).magnitude())
                for R_eq in equivalent_orientations(R_target, fold_data))
    return position_mm, float(np.degrees(angle))


def _sample(names, settle_min_s, steady_s, timeout_s):
    import rclpy
    from rclpy.node import Node
    from tf2_msgs.msg import TFMessage

    rclpy.init()
    node = Node("sim_seat_scorer")
    tape = []          # (stamp_s, {name: T})

    def callback(message):
        poses, stamp = {}, None
        for transform in message.transforms:
            stamp = transform.header.stamp.sec + transform.header.stamp.nanosec * 1e-9
            t, r = transform.transform.translation, transform.transform.rotation
            poses[transform.child_frame_id] = _matrix([t.x, t.y, t.z], [r.x, r.y, r.z, r.w])
        if stamp is not None:
            tape.append((stamp, poses))

    node.create_subscription(TFMessage, TOPIC, callback, 10)
    deadline = time.monotonic() + timeout_s
    first_stamp = None
    steady = False
    try:
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
            if not tape:
                continue
            first_stamp = tape[0][0] if first_stamp is None else first_stamp
            latest = tape[-1][0]
            if latest - first_stamp < max(settle_min_s, steady_s):
                continue
            window = [poses for stamp, poses in tape if stamp >= latest - steady_s]
            if any(name not in window[-1] for name in names):
                continue
            steady = True
            for name in names:
                rows = [poses[name] for poses in window if name in poses]
                end = rows[-1]
                for T in rows:
                    if (np.linalg.norm(T[:3, 3] - end[:3, 3]) * 1000.0 > STEADY_MM or
                            np.degrees(R.from_matrix(T[:3, :3].T @ end[:3, :3]).magnitude()) > STEADY_DEG):
                        steady = False
                        break
            if steady:
                break
    finally:
        node.destroy_node()
        rclpy.shutdown()
    if not tape:
        return None, None, False
    return tape[-1][1], (tape[0][0], tape[-1][0]), steady


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--score", default="")
    ap.add_argument("--observe", default="")
    ap.add_argument("--drop", default="")
    ap.add_argument("--base", default="base1")
    ap.add_argument("--settle-min-s", type=float, default=0.5)
    ap.add_argument("--steady-s", type=float, default=0.3)
    ap.add_argument("--timeout-s", type=float, default=20.0)
    args = ap.parse_args(argv)
    if os.environ.get("ROS_MCP_MODE") != "sim":
        print("__RESULT_JSON__\n" + json.dumps({"result": "failure",
              "error": "sim_seat_scorer reads simulator ground truth; it refuses outside ROS_MCP_MODE=sim"})
              + "\n__END_RESULT_JSON__")
        return 2
    names = [n for n in (args.score or args.observe or args.drop).split(",") if n]
    if not names or sum(map(bool, (args.score, args.observe, args.drop))) != 1:
        ap.error("give exactly one of --score, --observe or --drop with part names")
    poses, span, steady = _sample(names + ([args.base] if args.score else []),
                                  args.settle_min_s if args.score else 0.0,
                                  args.steady_s if args.score else 0.0, args.timeout_s)
    result = {"result": "failure", "mode": "sim", "base": args.base}
    if poses is None:
        result["error"] = f"no {TOPIC} sample within {args.timeout_s} s"
    elif args.drop:
        from primitives.shared.config import TABLE_COLLISION_MARGIN_SIDEWAYS, TABLE_HEIGHT
        from primitives.shared.held_part_collision import part_surface_points
        parts = {}
        for name in names:
            if name not in poses:
                parts[name] = {"error": "pose missing"}
                continue
            P = part_surface_points(name) @ poses[name][:3, :3].T + poses[name][:3, 3]
            drop = float(np.min(P[:, 2]) - TABLE_HEIGHT)
            parts[name] = {"drop_mm": round(1000.0 * drop, 2),
                           "within_floor_bound": drop <= TABLE_COLLISION_MARGIN_SIDEWAYS}
        ok = all(p.get("within_floor_bound") for p in parts.values())
        result.update({"result": "success" if ok else "failure", "parts": parts,
                       "bound_mm": 1000.0 * TABLE_COLLISION_MARGIN_SIDEWAYS,
                       "bound_source": "real place-down floor: grasp point TABLE_COLLISION_MARGIN_SIDEWAYS above "
                                       "the table for a sideways stance (replay_real_assembly._move_down_floor_z)"})
        if not ok:
            result["error"] = "SIM_PLACE_DOWN_DROP_EXCEEDS_FLOOR: " + "; ".join(
                f"{n} {p.get('drop_mm')} mm above the table" for n, p in parts.items() if not p.get("within_floor_bound"))
    elif args.observe:
        missing = [n for n in names if n not in poses]
        result.update({"result": "failure" if missing else "success", "missing": missing,
                       "parts": {n: {"xyz": poses[n][:3, 3].tolist(),
                                     "quat_xyzw": R.from_matrix(poses[n][:3, :3]).as_quat().tolist()}
                                 for n in names if n in poses},
                       "sim_time_s": span[1]})
    else:
        from primitives.shared.fold_symmetry import load_symmetry_data
        from utils.data_path_finder import (find_assembly_json_by_base_name, get_assembly_data_dir,
                                            get_symmetry_dir)
        components = {c["name"]: c for c in json.load(open(find_assembly_json_by_base_name(
            args.base, str(get_assembly_data_dir()))))["components"]}
        parts = {}
        for name in names:
            if name not in poses or args.base not in poses:
                parts[name] = {"seated": False, "error": "pose missing"}
                continue
            mm, deg = seat_error(poses[name], poses[args.base], components[name],
                                 load_symmetry_data(name, str(get_symmetry_dir())))
            parts[name] = {"seat_mm": round(mm, 3), "seat_deg": round(deg, 3),
                           "seated": mm <= SEAT_BAR_MM and deg <= SEAT_BAR_DEG}
        ok = steady and all(p["seated"] for p in parts.values())
        result.update({"result": "success" if ok else "failure", "parts": parts, "settled": steady,
                       "sim_window_s": list(span), "bar": {"mm": SEAT_BAR_MM, "deg": SEAT_BAR_DEG},
                       "ruling": RULING})
        if not steady:
            result["error"] = "parts did not settle within the timeout; no seat verdict"
        elif not ok:
            result["error"] = "; ".join(f"{n} not seated: {p.get('seat_mm')} mm / {p.get('seat_deg')} deg"
                                        for n, p in parts.items() if not p["seated"])
    print("__RESULT_JSON__\n" + json.dumps(result, sort_keys=True) + "\n__END_RESULT_JSON__", flush=True)
    return 0 if result["result"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
